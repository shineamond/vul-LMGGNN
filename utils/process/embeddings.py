import numpy as np
import torch
from torch_geometric.data import Data
from utils.functions import tokenizer
from utils.functions import log as logger
from gensim.models.keyedvectors import Word2VecKeyedVectors
from models.layers import encode_input
from transformers import RobertaTokenizer, RobertaModel


def _norm_edge_type(t: str) -> str:
    return (t or "").strip().lower()


def _edge_type_set(edge_type):
    """
    Normalizes the edge_type configuration.

    - None / "Cpg" / "CPG" / "all" / "*" => accept all edge types
    - "Ast" => accept only Ast (case-insensitive)
    - ["Ast", "Cfg"] => accept Ast + Cfg
    """
    if edge_type is None:
        return None
    if isinstance(edge_type, (list, tuple, set)):
        s = {_norm_edge_type(x) for x in edge_type if x is not None}
        return s if len(s) > 0 else None

    et = _norm_edge_type(str(edge_type))
    if et in ("cpg", "all", "*", "any"):
        return None
    return {et}


def _compute_non_leaf_ids(nodes, ast_edge_types=("ast",)):
    """
    Identifies *non-leaf* AST nodes.

    In Joern JSON exports, AST edges are typically stored as child -> parent
    (edge.in = child, edge.out = parent). A node is considered non-leaf if it
    appears as a parent of at least one AST edge.
    """
    ast_set = {_norm_edge_type(x) for x in ast_edge_types}
    parents = set()

    for n_id, node in nodes.items():
        for edge in node.edges.values():
            if _norm_edge_type(edge.type) in ast_set:
                if edge.node_out in nodes and edge.node_out != n_id:
                    parents.add(edge.node_out)

    return parents


class NodesEmbedding:
    def __init__(self, nodes_dim: int, w2v_keyed_vectors: Word2VecKeyedVectors):
        self.nodes_dim = nodes_dim
        assert self.nodes_dim >= 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer_bert = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.bert_model = RobertaModel.from_pretrained("microsoft/codebert-base").to(self.device)
        self.bert_model.eval()

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
        non_leaf_ids = _compute_non_leaf_ids(nodes, ast_edge_types=("ast", "AST"))

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
        self.edge_types = _edge_type_set(edge_type)

    def __call__(self, nodes):
        connections = self.nodes_connectivity(nodes)

        return torch.tensor(connections).long()
    
    def _accept(self, edge_type: str) -> bool:
        if self.edge_types is None:
            return True
        return _norm_edge_type(edge_type) in self.edge_types

    # nodesToGraphConnectivity
    def nodes_connectivity(self, nodes):
        # nodes are ordered by line and column
        coo = [[], []]

        for _, node in nodes.items():
            for edge in node.edges.values():
                if not self._accept(edge.type):
                    continue

                if edge.node_in in nodes and edge.node_out in nodes:
                    src = nodes[edge.node_in].order
                    dst = nodes[edge.node_out].order
                    if src == dst:
                        continue

                    # Bidirectional edge index (common in GGNN settings)
                    coo[0].append(src); coo[1].append(dst)
                    coo[0].append(dst); coo[1].append(src)

        return coo


def nodes_to_input(nodes, target, nodes_dim, edge_type, keyed_vectors = None):
    nodes_embedding = NodesEmbedding(nodes_dim, keyed_vectors)
    graphs_embedding = GraphsEmbedding(edge_type)
    label = torch.tensor([target]).float()

    return Data(x=nodes_embedding(nodes), edge_index=graphs_embedding(nodes), y=label)
