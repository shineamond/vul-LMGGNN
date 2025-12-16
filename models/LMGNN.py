import torch as th
import torch.nn as nn
import torch.nn.functional as F
from models.layers import Conv, encode_input
from torch_geometric.nn.conv import GatedGraphConv
from transformers import AutoModel, AutoTokenizer
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import numpy as np


def _is_non_leaf_ast(node, nodes, ast_edge_type: str = "AST") -> bool:
    for _, edge in node.edges.items():
        if edge.type != ast_edge_type:
            continue
        if edge.node_out in nodes and edge.node_out != node.id:
            return True
    return False


class BertGGCN(nn.Module):
    def __init__(self, gated_graph_conv_args, conv_args, emb_size, device):
        super(BertGGCN, self).__init__()
        self.k = 0.1
        self.ggnn = GatedGraphConv(**gated_graph_conv_args).to(device)
        self.conv = Conv(**conv_args,
                         fc_1_size=gated_graph_conv_args["out_channels"] + emb_size,
                         fc_2_size=gated_graph_conv_args["out_channels"]).to(device)
        self.nb_class = 2
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.bert_model = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, self.nb_class).to(device)
        self.device = device
        # self.conv.apply(init_weights)

    def forward(self, data):
        # the DataLoader format
        # DataBatch(x=[1640, 101], edge_index=[2, 933], y=[8], func=[8], batch=[1640], ptr=[9])

        if self.training:
            self.update_nodes(data)

        x, edge_index, text = data.x, data.edge_index, data.func
        x = self.ggnn(x, edge_index)
        p_ggcn = self.conv(x, data.x)

        input_ids, attention_mask = encode_input(text, self.tokenizer)
        cls_feats = self.bert_model(input_ids.to(self.device), attention_mask.to(self.device))[0][:, 0]
        cls_logit = self.classifier(cls_feats.to(self.device))
        logit_ggcn = th.log(p_ggcn + 1e-12)

        combined_logit = self.k * logit_ggcn + (1 - self.k) * cls_logit
        pred = F.softmax(combined_logit, dim = 1)
        pred = pred.clamp(min = 1e-12, max = 1.0 - 1e-12)

        return pred
    
    def forward_with_node_embeddings(self, data):
        if self.training:
            self.update_nodes(data)

        x, edge_index, text = data.x, data.edge_index, data.func
        node_emb = self.ggnn(x, edge_index)
        p_ggcn = self.conv(node_emb, data.x)

        input_ids, attention_mask = encode_input(text, self.tokenizer)
        cls_feats = self.bert_model(
            input_ids.to(self.device),
            attention_mask.to(self.device)
        )[0][:, 0]
        cls_logit = self.classifier(cls_feats.to(self.device))
        logit_ggcn = th.log(p_ggcn + 1e-12)
        
        combined_logit = self.k * logit_ggcn + (1 - self.k) * cls_logit
        pred = F.softmax(combined_logit, dim=1)
        pred = pred.clamp(min = 1e-12, max = 1.0 - 1e-12)

        return pred, node_emb


    def update_nodes(self, data):

        if not isinstance(data.x, dict):
            return
        
        nodes = data.x
        ordered = sorted(nodes.items(), key=lambda kv: (kv[1].order if kv[1].order is not None else 10**9))
        non_leaf_ids = {n_id for n_id, n in nodes.items() if _is_non_leaf_ast(n, nodes, ast_edge_type="AST")}

        embeddings = []
        with th.no_grad():
            for n_id, node in ordered:
                node_code = "" if n_id in non_leaf_ids else (node.get_code() or "")
                input_ids, attention_mask = encode_input(node_code, self.tokenizer, max_length=128)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                token_emb = self.bert_model.embeddings.word_embeddings(input_ids)  # (1, L, H)
                mask = attention_mask.unsqueeze(-1).float()
                summed = (token_emb * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                source_embedding = (summed / denom).squeeze(0).cpu().numpy()

                embedding = np.concatenate((np.array([node.type], dtype=np.float32), source_embedding.astype(np.float32)), axis=0)
                embeddings.append(embedding)

        data.x = th.tensor(np.array(embeddings, dtype=np.float32), dtype=th.float32, device=self.device)

    def save(self, path):
        print(path)
        th.save(self.state_dict(), path)
        print("save!!!!!!")

    def load(self, path):
        self.load_state_dict(th.load(path))

