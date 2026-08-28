import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(in_channels, out_channels),
                ReLU(),
                Linear(out_channels, out_channels),
            )
        )
        self.conv2 = GINConv(
            Sequential(
                Linear(out_channels, out_channels),
                ReLU(),
                Linear(out_channels, out_channels),
            )
        )
        self.attention = GraphAttentionShortcut(in_channels, out_channels)

    def forward(self, x, edge_index, batch, states, rnn):
        identity = x
        out = F.relu(self.conv1(x, edge_index))
        out = self.conv2(out, edge_index)

        out = self.attention(out, identity, batch)


        out_detach = out.detach()
        rnn_out, (hidden_state, cell_state) = rnn(out_detach[None, :, :], states)
        rnn_out = rnn_out.squeeze()

        return out, rnn_out, (hidden_state, cell_state)

class GraphAttentionShortcut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels != out_channels:
            self.proj = Sequential(
                Linear(in_channels, out_channels),
                ReLU()
            )
        else:
            self.proj = nn.Identity()

        self.attn = Sequential(
            Linear(out_channels, out_channels),
            nn.Tanh(),
            Linear(out_channels, 1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, current, shortcut, batch):
        shortcut = self.proj(shortcut)
        attn_weights = self.attn(current)
        attended_shortcut = shortcut * attn_weights
        out = current + attended_shortcut
        return self.norm(out)

class ResGINAtt(nn.Module):
    def __init__(
            self,
            molecule_channels: int = 78,
            hidden_channels: int = 128,
            middle_channels: int = 64,
            layer_count: int = 2,
            out_channels: int = 2,
            dropout_rate: int = 0.2
    ):
        super().__init__()
        self.layer_count = layer_count
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(molecule_channels if i == 0 else hidden_channels, hidden_channels)
            for i in range(layer_count)
        ])
        self.border_rnn = nn.LSTM(hidden_channels, hidden_channels, 1)
        self.final = Sequential(
            Linear(4 * hidden_channels + 256, middle_channels),
            ReLU(),
            Linear(middle_channels, out_channels),
        )
        self.reduction = Sequential(
            Linear(954, 2048),
            ReLU(),
            nn.Dropout(dropout_rate),
            Linear(2048, 512),
            ReLU(),
            nn.Dropout(dropout_rate),
            Linear(512, 256),
            ReLU()
        )
        self.reduction2 = Sequential(
            Linear(954, 2048),
            ReLU(),
            nn.Dropout(dropout_rate),
            Linear(2048, 512),
            ReLU(),
            nn.Dropout(dropout_rate),
            Linear(512, 78),
            ReLU()
        )
        self.pool1 = Attention(hidden_channels, 4)
        self.pool2 = Attention(hidden_channels, 4)

    def forward(self, molecules_left, molecules_right) -> torch.FloatTensor:
        x1, edge_index1, batch1, cell, mask1 = molecules_left.x, molecules_left.edge_index, molecules_left.batch, molecules_left.cell, molecules_left.mask
        x2, edge_index2, batch2, mask2 = molecules_right.x, molecules_right.edge_index, molecules_right.batch, molecules_right.mask

        cell = F.normalize(cell, 2, 1)
        cell_expand = self.reduction2(cell)
        cell = self.reduction(cell)
        cell_expand = cell_expand.unsqueeze(1).expand(cell.shape[0], 100, -1).reshape(-1, 78)

        batch_size = torch.max(molecules_left.batch) + 1
        mask1 = mask1.reshape((batch_size, 100))
        mask2 = mask2.reshape((batch_size, 100))

        left_states, right_states = None, None
        gcn_hidden_left = molecules_left.x + cell_expand
        gcn_hidden_right = molecules_right.x + cell_expand

        for block in self.residual_blocks:
            gcn_hidden_left, rnn_out_left, left_states = block(
                gcn_hidden_left, edge_index1, batch1, left_states, self.border_rnn
            )
            gcn_hidden_right, rnn_out_right, right_states = block(
                gcn_hidden_right, edge_index2, batch2, right_states, self.border_rnn
            )

        rnn_out_left = rnn_out_left.reshape(batch_size, 100, -1)
        rnn_out_right = rnn_out_right.reshape(batch_size, 100, -1)
        rnn_pooled_left, rnn_pooled_right = self.pool1(rnn_out_left, rnn_out_right, (mask1, mask2))

        gcn_hidden_left = gcn_hidden_left.reshape(batch_size, 100, -1)
        gcn_hidden_right = gcn_hidden_right.reshape(batch_size, 100, -1)
        gcn_hidden_left, gcn_hidden_right = self.pool2(gcn_hidden_left, gcn_hidden_right, (mask1, mask2))

        shared_graph_level = torch.cat([gcn_hidden_left, gcn_hidden_right], dim=1)
        out = torch.cat([shared_graph_level, rnn_pooled_left, rnn_pooled_right, cell], dim=1)
        return self.final(out)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim_per_head = dim // num_heads
        self.linear_q = Linear(dim, self.dim_per_head * num_heads)
        self.linear_k = Linear(dim, self.dim_per_head * num_heads)
        self.linear_v = Linear(dim, self.dim_per_head * num_heads)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=0.2)

    def attention(self, q1, k1, v1, q2, k2, v2, attn_mask=None):
        a1 = torch.tanh(torch.bmm(k1, q2.transpose(1, 2)))
        a2 = torch.tanh(torch.bmm(k2, q1.transpose(1, 2)))

        if attn_mask is not None:
            mask1, mask2 = attn_mask
            a1 = torch.softmax(torch.sum(a1, dim=2).masked_fill(mask1, -np.inf), dim=-1).unsqueeze(dim=1)
            a2 = torch.softmax(torch.sum(a2, dim=2).masked_fill(mask2, -np.inf), dim=-1).unsqueeze(dim=1)
        else:
            a1 = torch.softmax(torch.sum(a1, dim=2), dim=1).unsqueeze(dim=1)
            a2 = torch.softmax(torch.sum(a2, dim=2), dim=1).unsqueeze(dim=1)

        a1 = self.dropout(a1)
        a2 = self.dropout(a2)
        vector1 = torch.bmm(a1, v1).squeeze()
        vector2 = torch.bmm(a2, v2).squeeze()
        return vector1, vector2

    def forward(self, fingerprint_vectors1, fingerprint_vectors2, attn_mask=None):
        q1, q2 = F.relu(self.linear_q(fingerprint_vectors1)), F.relu(self.linear_q(fingerprint_vectors2))
        k1, k2 = F.relu(self.linear_k(fingerprint_vectors1)), F.relu(self.linear_k(fingerprint_vectors2))
        v1, v2 = F.relu(self.linear_v(fingerprint_vectors1)), F.relu(self.linear_v(fingerprint_vectors2))
        vector1, vector2 = self.attention(q1, k1, v1, q2, k2, v2, attn_mask)
        vector1 = self.norm(torch.mean(fingerprint_vectors1, dim=1) + vector1)
        vector2 = self.norm(torch.mean(fingerprint_vectors2, dim=1) + vector2)
        return vector1, vector2