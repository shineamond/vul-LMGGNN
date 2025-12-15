import numpy as np
import torch
from torch_geometric.data import Data
from utils.functions import tokenizer
from utils.functions import log as logger
from gensim.models.keyedvectors import Word2VecKeyedVectors
from models.layers import encode_input
from transformers import RobertaTokenizer, RobertaModel

def _is_non_leaf_ast(node, nodes, ast_edge_type: str = "AST") -> bool:
    """A conservative 'non-leaf' check: any outgoing AST edge to an existing node."""
    for _, edge in node.edges.items():
        if edge.type != ast_edge_type:
            continue
        # If this node has an outgoing AST edge, it is an internal node.
        if edge.node_out in nodes and edge.node_out != node.id:
            return True
    return False

class NodesEmbedding:
    def __init__(self, nodes_dim: int, w2v_keyed_vectors: Word2VecKeyedVectors):
        self.nodes_dim = nodes_dim
        assert self.nodes_dim >= 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer_bert = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.bert_model = RobertaModel.from_pretrained("microsoft/codebert-base").to(self.device)

        self.feat_dim = self.bert_model.config.hidden_size
        self.kv_size = self.feat_dim        

        # Buffer for embeddings with padding
        self.target = torch.zeros(self.nodes_dim, self.kv_size + 1).float()

        self.w2v_keyed_vectors = w2v_keyed_vectors

    def __call__(self, nodes):
        target = torch.zeros(self.nodes_dim, self.kv_size + 1, dtype=torch.float32)
        embedded_nodes = self.embed_nodes(nodes)
        nodes_tensor = torch.from_numpy(embedded_nodes).float()

        n = min(nodes_tensor.size(0), target.size(0))
        target[:n, :] = nodes_tensor[:n, :]

        return target

    @torch.no_grad()
    def embed_nodes(self, nodes):
        non_leaf_ids = {n_id for n_id, n in nodes.items() if _is_non_leaf_ast(n, nodes, ast_edge_type="AST")}

        embeddings = []
        for n_id, node in nodes.items():
            # Get node's code
            node_code = "" if n_id in non_leaf_ids else (node.get_code() or "")
            input_ids, attention_mask = encode_input(node_code, self.tokenizer_bert)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            token_emb = self.bert_model.embeddings.word_embeddings(input_ids)  # (1, L, H)

            mask = attention_mask.unsqueeze(-1).float()  # (1, L, 1)
            summed = (token_emb * mask).sum(dim=1)       # (1, H)
            denom = mask.sum(dim=1).clamp(min=1.0)       # (1, 1)
            source_embedding = (summed / denom).squeeze(0).cpu().numpy()  # (H,)

            embedding = np.concatenate((np.array([node.type], dtype=np.float32), source_embedding.astype(np.float32)), axis=0)
            embeddings.append(embedding)
        # print(node.label, node.properties.properties.get("METHOD_FULL_NAME"))

        return np.array(embeddings, dtype=np.float32)

    # fromTokenToVectors
    # This is the original Word2Vec model usage.
    # Although we keep this part of the code, we are not using it.
    def get_vectors(self, tokenized_code, node):
        vectors = []

        for token in tokenized_code:
            if token in self.w2v_keyed_vectors.vocab:
                vectors.append(self.w2v_keyed_vectors[token])
            else:
                # print(node.label, token, node.get_code(), tokenized_code)
                vectors.append(np.zeros(self.kv_size))
                if node.label not in ["Identifier", "Literal", "MethodParameterIn", "MethodParameterOut"]:
                    msg = f"No vector for TOKEN {token} in {node.get_code()}."
                    logger.log_warning('embeddings', msg)

        return vectors


class GraphsEmbedding:
    def __init__(self, edge_type):
        self.edge_type = edge_type

    def __call__(self, nodes):
        connections = self.nodes_connectivity(nodes)

        return torch.tensor(connections).long()

    # nodesToGraphConnectivity
    def nodes_connectivity(self, nodes):
        # nodes are ordered by line and column
        coo = [[], []]

        for node_idx, (node_id, node) in enumerate(nodes.items()):
            if node_idx != node.order:
                raise Exception("Something wrong with the order")

            for e_id, edge in node.edges.items():
                if edge.type != self.edge_type:
                    continue

                if edge.node_in in nodes and edge.node_in != node_id:
                    coo[0].append(nodes[edge.node_in].order)
                    coo[1].append(node_idx)

                if edge.node_out in nodes and edge.node_out != node_id:
                    coo[0].append(node_idx)
                    coo[1].append(nodes[edge.node_out].order)

        return coo


def nodes_to_input(nodes, target, nodes_dim, edge_type, keyed_vectors = None):
    nodes_embedding = NodesEmbedding(nodes_dim, keyed_vectors)
    graphs_embedding = GraphsEmbedding(edge_type)
    label = torch.tensor([target]).float()

    return Data(x=nodes_embedding(nodes), edge_index=graphs_embedding(nodes), y=label)
