import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CustomConfig


def generate_casual_mask(batch_size, num_heads, seq_len):
    # 创建基础掩码（下三角矩阵）
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    # 扩展掩码到批次+ num_heads维度
    mask = mask.unsqueeze(0).expand(batch_size, num_heads, -1, -1)
    return mask


def create_sinusoidal_embeddings(max_seq_len, d_model):
    position = torch.arange(0, max_seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe = torch.zeros(max_seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return nn.Parameter(pe.unsqueeze(0), requires_grad=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin):
    q_embed = (q * freqs_cos) + (rotate_half(q) * freqs_sin)
    k_embed = (k * freqs_cos) + (rotate_half(k) * freqs_sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions):
        # positions: [batch_size, seq_len] or [batch_size, num_heads, seq_len]
        if positions.dim() == 3:
            positions = positions[:, 0, :]  # 只使用第一个 head 的位置，因为所有 head 的位置相同

        freqs = torch.einsum("bi,j->bij", positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None, alibi=None, freqs_cos=None, freqs_sin=None):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if freqs_cos is not None and freqs_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask
        if alibi is not None:
            scores += alibi

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)
        self.attention_norm = RMSNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_norm = RMSNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None, alibi=None, freqs_cos=None, freqs_sin=None):
        # Pre-norm for attention
        normed_x = self.attention_norm(x)
        attention_output = self.attention(normed_x, attention_mask, alibi, freqs_cos, freqs_sin)
        x = x + self.dropout(attention_output)

        # Pre-norm for feed-forward
        normed_x = self.ffn_norm(x)
        ff_output = self.feed_forward(normed_x)
        x = x + self.dropout(ff_output)

        return x


class CustomTransformerBase(nn.Module):

    def __init__(self, cfg: CustomConfig):
        super().__init__()
        self.pad_token_id = 0
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg.hidden_size, cfg.num_heads, cfg.hidden_size * 4, cfg.dropout, cfg.layer_norm_eps)
             for _ in range(cfg.num_layers)]
        )
        self.fc = nn.Linear(cfg.hidden_size, cfg.token_vocab_size)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        # Embedding
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.dropout(x)
        # Generate custom attention mask
        if attention_mask is None:
            attention_mask = generate_casual_mask(batch_size, self.num_heads, seq_len).to(x.device)
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
