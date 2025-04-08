# import torch
# import torch.nn as nn
# from torch_geometric import nn as PyG
# from transformers import EsmModel  # 修改为ESM模型
# import logging
# from transformers import logging as transformers_logging
#
# transformers_logging.set_verbosity_error()
#
#
# class GNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(GNN, self).__init__()
#         self.gnn1 = PyG.SAGEConv(input_dim, 128)
#         self.gnn2 = PyG.SAGEConv(128, 256)
#         self.transformer1 = PyG.TransformerConv(256, 32, heads=8)
#         self.transformer2 = PyG.TransformerConv(256, 32, heads=8)
#         self.FC = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, hidden_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.gnn1(x, edge_index)
#         x = torch.relu(x)
#         x = self.gnn2(x, edge_index)
#         x = torch.relu(x)
#         x = self.transformer1(x, edge_index)
#         x = torch.relu(x)
#         x = self.transformer2(x, edge_index)
#         x = torch.relu(x)
#         return PyG.global_max_pool(x, data.batch)
#
#
# class PeptideESM(nn.Module):  # 修改类名和实现
#     def __init__(self):
#         super(PeptideESM, self).__init__()
#         self.esm = EsmModel.from_pretrained(
#             "/mnt/sdb/cjj/MultiPeptide-main/esm-2",
#             local_files_only=True
#         )
#
#     def forward(self, inputs, attention_mask):
#         output = self.esm(inputs, attention_mask=attention_mask)
#         return output.last_hidden_state[:, 0, :]  # 使用CLS token作为特征
#
#
# class ProjectionHead(nn.Module):
#     def __init__(self, embedding_dim, projection_dim, dropout):
#         super(ProjectionHead, self).__init__()
#         self.projection = nn.Linear(embedding_dim, projection_dim)
#         self.gelu = nn.GELU()
#         self.fc = nn.Linear(projection_dim, projection_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(projection_dim)
#
#     def forward(self, x):
#         projected = self.projection(x)
#         x = self.gelu(projected)
#         x = self.fc(x)
#         x = self.dropout(x)
#         x = x + projected
#         return self.layer_norm(x)
#
#
# class PretrainNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim, projection_dim, dropout):
#         super(PretrainNetwork, self).__init__()
#         self.gnn = GNN(input_dim, hidden_dim)
#         self.esm = PeptideESM()
#
#
#         esm_hidden_size = self.esm.config.hidden_size
#
#         self.graph_projection = ProjectionHead(hidden_dim, projection_dim, dropout)
#         self.text_projection = ProjectionHead(esm_hidden_size, projection_dim, dropout)  # 使用动态维度
#
#     def forward(self, data):
#         gnn_features = self.gnn(data)
#         esm_features = self.esm(data.seq, data.attn_mask)
#         gnn_embs = self.graph_projection(gnn_features)
#         esm_embs = self.text_projection(esm_features)
#         gnn_embs = gnn_embs / torch.linalg.norm(gnn_embs, dim=1, keepdim=True)
#         esm_embs = esm_embs / torch.linalg.norm(esm_embs, dim=1, keepdim=True)
#         return esm_embs, gnn_embs
#
#
# class CLIPLoss(nn.Module):
#     def __init__(self, temperature):
#         super(CLIPLoss, self).__init__()
#         self.temperature = temperature
#         self.logsoftmax = nn.LogSoftmax(dim=-1)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, esm_embs, gnn_embs, labels):
#         gnn_sim = gnn_embs @ gnn_embs.T
#         esm_sim = esm_embs @ esm_embs.T
#         targets = self.softmax((gnn_sim + esm_sim) / (2 * self.temperature))
#         logits = (esm_embs @ gnn_embs.T) / self.temperature
#         gnn_loss = self.cross_entropy(logits.T, targets.T)
#         esm_loss = self.cross_entropy(logits, targets)
#         return (gnn_loss + esm_loss).mean() / 2
#
#     def cross_entropy(self, logits, targets):
#         return (-targets * self.logsoftmax(logits)).sum(1)
#
#
# def create_model(config, get_embeddings=False):
#     model = PretrainNetwork(
#         input_dim=config['network']['GNN']['input_dim'],
#         hidden_dim=config['network']['GNN']['hidden_dim'],
#         projection_dim=config['network']['proj_dim'],
#         dropout=config['network']['drp']
#     ).to(config['device'])
#     return model
#
#
# def cri_opt_sch(config, model):
#     criterion = CLIPLoss(1)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=config['optim']['lr'])
#
#     if config['sch']['name'] == 'onecycle':
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(
#             optimizer,
#             max_lr=config['optim']['lr'],
#             epochs=config['epochs'],
#             steps_per_epoch=config['sch']['steps']
#         )
#     elif config['sch']['name'] == 'lronplateau':
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             factor=config['sch']['factor'],
#             patience=config['sch']['patience']
#         )
#     return criterion, optimizer, scheduler


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric import nn as PyG
from transformers import BertModel, BertConfig, logging
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()

import torch.nn.functional as F
import torch_geometric.nn as PyG


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNN, self).__init__()

        # 改进的GAT特征提取层（增加深度和多头注意力）
        self.gat1 = PyG.GATConv(input_dim, 32, heads=8, dropout=0.3)  # 8头注意力
        self.gat2 = PyG.GATConv(8 * 32, 64, heads=8, dropout=0.3)  # 层级间维度扩展
        self.gat3 = PyG.GATConv(8 * 64, 256, heads=4, concat=False)  # 最终统一维度

        # 增强的全局交互层（结合残差连接）
        self.transformer1 = PyG.TransformerConv(256, 64, heads=8, dropout=0.3)
        self.transformer2 = PyG.TransformerConv(64 * 8, 256, heads=4, concat=False)

        # 优化后的特征映射层
        self.FC = torch.nn.Sequential(
            torch.nn.LayerNorm(512),
            torch.nn.Linear(512, 512),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, hidden_dim),
            torch.nn.Sigmoid()
        )

        # 正则化组件
        self.bn1 = torch.nn.BatchNorm1d(8 * 32)
        self.bn2 = torch.nn.BatchNorm1d(8 * 64)
        self.bn3 = torch.nn.BatchNorm1d(256)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # GAT局部特征提取（带层级规范化）
        x = F.leaky_relu(self.bn1(self.gat1(x, edge_index)), 0.2)
        x = F.leaky_relu(self.bn2(self.gat2(x, edge_index)), 0.2)
        x = F.leaky_relu(self.bn3(self.gat3(x, edge_index)), 0.2)

        # 全局交互层（带残差连接）
        identity = x
        x = F.leaky_relu(self.transformer1(x, edge_index), 0.2)
        x = F.leaky_relu(self.transformer2(x, edge_index), 0.2)
        x += identity  # 残差连接防止梯度消失

        # 改进的池化策略（混合最大+平均池化）
        max_pool = PyG.global_max_pool(x, data.batch)
        mean_pool = PyG.global_mean_pool(x, data.batch)
        x = torch.cat([max_pool, mean_pool], dim=1)

        return self.FC(x)

# class GNN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(GNN, self).__init__()
#         # 局部特征提取层（GNN）
#         self.gnn1 = PyG.nn.SAGEConv(input_dim, 128)
#         self.gnn2 = PyG.nn.SAGEConv(128, 256)
#
#         # 全局交互层（Graph Transformer）
#         self.transformer1 = PyG.nn.TransformerConv(256, 32, heads=8)
#         self.transformer2 = PyG.nn.TransformerConv(256, 32, heads=8)
#
#         # 特征映射层
#         self.FC = torch.nn.Sequential(
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, 512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, hidden_dim),
#             torch.nn.Sigmoid()
#         )
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         # GNN局部特征
#         x = self.gnn1(x, edge_index)
#         x = torch.relu(x)
#         x = self.gnn2(x, edge_index)
#         x = torch.relu(x)
#
#         # Transformer全局交互
#         x = self.transformer1(x, edge_index)
#         x = torch.relu(x)
#         x = self.transformer2(x, edge_index)
#         x = torch.relu(x)
#
#         # 投影到目标维度
#         x = self.FC(x)
#
#         return PyG.nn.global_max_pool(x, data.batch)


class PeptideBERT(torch.nn.Module):
    def __init__(self, bert_config):
        super(PeptideBERT, self).__init__()
        # 加载本地模型文件
        local_model_path = "/mnt/sdb/cjj/MultiPeptide-main/prot_bert_bfd"  # 使用 Linux 路径格式
        self.protbert = BertModel.from_pretrained(
            local_model_path,  # 替换为本地模型路径
            config=bert_config,
            ignore_mismatched_sizes=True
        )

    def forward(self, inputs, attention_mask):
        output = self.protbert(inputs, attention_mask=attention_mask)

        return output.pooler_output


class ProjectionHead(torch.nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super(ProjectionHead, self).__init__()

        self.projection = torch.nn.Linear(embedding_dim, projection_dim)
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(projection_dim, projection_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class PretrainNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, bert_config, projection_dim, dropout):
        super(PretrainNetwork, self).__init__()

        self.gnn = GNN(input_dim, hidden_dim)
        self.bert = PeptideBERT(bert_config)
        self.graph_projection = ProjectionHead(hidden_dim, projection_dim, dropout)
        self.text_projection = ProjectionHead(bert_config.hidden_size, projection_dim, dropout)

    def forward(self, data):
        gnn_features = self.gnn(data)
        bert_features = self.bert(data.seq, data.attn_mask)

        gnn_embs = self.graph_projection(gnn_features)
        bert_embs = self.text_projection(bert_features)

        gnn_embs = gnn_embs / torch.linalg.norm(gnn_embs, dim=1, keepdim=True)
        bert_embs = bert_embs / torch.linalg.norm(bert_embs, dim=1, keepdim=True)

        return bert_embs, gnn_embs

class CLIPLoss(torch.nn.Module):
    def __init__(self, temperature):
        super(CLIPLoss, self).__init__()
        self.temperature = temperature
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, bert_embs, gnn_embs, labels):
        gnn_sim = gnn_embs @ gnn_embs.T
        bert_sim = bert_embs @ bert_embs.T
        targets = self.softmax((gnn_sim + bert_sim) / (2 * self.temperature))
        logits = (bert_embs @ gnn_embs.T) / self.temperature

        gnn_loss = self.cross_entropy(logits.T, targets.T)
        bert_loss = self.cross_entropy(logits, targets)
        loss = (gnn_loss + bert_loss) / 2

        return loss.mean()

    def cross_entropy(self, logits, targets):
        log_probs = self.logsoftmax(logits)
        return (-targets * log_probs).sum(1)


def create_model(config, get_embeddings=False):
    bert_config = BertConfig(
        vocab_size=config['vocab_size'],
        hidden_size=config['network']['BERT']['hidden_size'],
        num_hidden_layers=config['network']['BERT']['hidden_layers'],
        num_attention_heads=config['network']['BERT']['attn_heads'],
        hidden_dropout_prob=config['network']['BERT']['dropout']
    )

    model = PretrainNetwork(
        input_dim=config['network']['GNN']['input_dim'],
        hidden_dim=config['network']['GNN']['hidden_dim'],
        bert_config=bert_config,
        projection_dim=config['network']['proj_dim'],
        dropout=config['network']['drp']
    ).to(config['device'])

    return model


def cri_opt_sch(config, model):
    criterion = CLIPLoss(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optim']['lr'])

    if config['sch']['name'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['optim']['lr'],
            epochs=config['epochs'],
            steps_per_epoch=config['sch']['steps']
        )
    elif config['sch']['name'] == 'lronplateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config['sch']['factor'],
            patience=config['sch']['patience']
        )

    return criterion, optimizer, scheduler