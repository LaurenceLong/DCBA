import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from config import CustomConfig
from positional_encoding import build_alibi_tensor


def generate_causal_mask(batch_size, num_heads, seq_len):
    # 创建基础掩码（下三角矩阵）
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    # 扩展掩码到批次+ num_heads维度
    mask = mask.unsqueeze(0).expand(batch_size, num_heads, -1, -1)
    return mask


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class DynamicContextualBias(nn.Module):
    def __init__(self, npos_max, head_dim, num_heads, mlp_width=32):
        super().__init__()
        self.npos_max = npos_max
        self.head_dim = head_dim
        self.num_heads = num_heads

        # CoPE部分
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim, npos_max))

        # DAPE部分
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_heads, mlp_width),
            nn.SiLU(),
            nn.Linear(mlp_width, num_heads)
        )

    def forward(self, query, attn_logits):
        # CoPE部分：计算连续位置
        gates = torch.sigmoid(attn_logits)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)

        # 插值
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor

        cope_bias = logits_ceil * w + logits_floor * (1 - w)

        # DAPE部分：动态调整
        cope_bias = rearrange(cope_bias, 'b h s1 s2 -> b s1 s2 h')
        attn_logits = rearrange(attn_logits, 'b h s1 s2 -> b s1 s2 h')

        combined = torch.cat((attn_logits, cope_bias), dim=-1)
        dynamic_bias = self.mlp(combined)

        dynamic_bias = rearrange(dynamic_bias, 'b s1 s2 h -> b h s1 s2')
        return dynamic_bias


class DCBAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1, npos_max=128):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // self.num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.contextual_bias_layer = DynamicContextualBias(npos_max, self.head_dim, num_heads)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Calculate dynamic contextual bias based on scores
        # bias = build_alibi_tensor(scores, scores.dtype)
        bias = self.contextual_bias_layer(q, scores)
        scores += bias

        if attention_mask is not None:
            scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(out)


class DCBATransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1, layer_norm_eps=1e-5, npos_max=128):
        super().__init__()
        self.attention = DCBAttention(hidden_size, num_heads, dropout, npos_max)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)
        self.attention_norm = RMSNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_norm = RMSNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # Pre-norm for attention
        normed_x = self.attention_norm(x)
        attention_output = self.attention(normed_x, attention_mask)
        x = x + self.dropout(attention_output)

        # Pre-norm for feed-forward
        normed_x = self.ffn_norm(x)
        ff_output = self.feed_forward(normed_x)
        x = x + self.dropout(ff_output)

        return x


class DCBATransformer(nn.Module):

    def __init__(self, cfg: CustomConfig):
        super().__init__()
        self.pad_token_id = 0
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [DCBATransformerBlock(cfg.hidden_size, cfg.num_heads, cfg.hidden_size * 4,
                                  cfg.dropout, cfg.layer_norm_eps, cfg.max_seq_len)
             for _ in range(cfg.num_layers)]
        )
        self.fc = nn.Linear(cfg.hidden_size, cfg.vocab_size)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        # Embedding
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.dropout(x)
        # Generate custom attention mask
        if attention_mask is None:
            attention_mask = generate_causal_mask(batch_size, self.num_heads, seq_len).to(x.device)
        # todo 添加位置编码
        # Process through transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask)
        # Output layer
        logits = self.fc(x)
        return logits

    def compute_loss(self, logits, targets):
        batch_size, seq_len, vocab_size = logits.shape

        # 将logits和targets展平
        flat_logits = logits.view(-1, vocab_size)
        flat_targets = targets.view(-1)

        # 计算交叉熵损失
        loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=self.pad_token_id)
        return loss

    def generate(self, x, max_new_tokens, eos_token_id, temperature=1.0, top_k=0, top_p=1.0, tokenizer=None):
        self.eval()
        if x.dim() == 1:
            x = x.unsqueeze(0)  # 添加批量维度

        batch_size = x.size(0)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(x)
                next_token_logits = logits[:, -1, :] / temperature

                for i in range(batch_size):
                    if top_k > 0:
                        indices_to_remove = next_token_logits[i] < torch.topk(next_token_logits[i], top_k)[0][-1]
                        next_token_logits[i][indices_to_remove] = -float('inf')

                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits[i], descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[i][indices_to_remove] = -float('inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                if tokenizer is not None:
                    print("next_token", tokenizer.decode(next_token), next_token)

                x = torch.cat([x, next_token.unsqueeze(1)], dim=1)

                if (next_token == eos_token_id).all():
                    break
        return x
