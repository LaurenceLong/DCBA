import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CustomConfig


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

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(out)


class ContextualBiasBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_bias_heads, intermediate_size, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_bias_heads, dropout=dropout)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)
        self.attention_norm = RMSNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_norm = RMSNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads

    def forward(self, x, attention_mask=None):
        # x shape: (batch_size, num_heads, seq_len, seq_len)
        batch_size, num_heads, seq_len, _ = x.size()

        # Reshape x to (batch_size * num_heads, seq_len, seq_len)
        x_reshaped = x.view(batch_size * num_heads, seq_len, seq_len)

        # Apply attention and feed-forward layers
        x_reshaped = self.attention_norm(x_reshaped)
        attention_output = self.attention(x_reshaped, attention_mask)
        x_reshaped = x_reshaped + self.dropout(attention_output)

        x_reshaped = self.ffn_norm(x_reshaped)
        ff_output = self.feed_forward(x_reshaped)
        x_reshaped = x_reshaped + self.dropout(ff_output)

        # Reshape back to (batch_size, num_heads, seq_len, seq_len)
        return x_reshaped.view(batch_size, num_heads, seq_len, seq_len)


class DCBAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_bias_layers, num_bias_heads, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // self.num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.contextual_bias_layers = nn.ModuleList(
            [ContextualBiasBlock(hidden_size, num_heads, num_bias_heads, hidden_size * 4, dropout=dropout,
                                 layer_norm_eps=layer_norm_eps)
             for _ in range(num_bias_layers)]
        )

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Calculate dynamic contextual bias based on scores
        bias = scores
        for bias_layer in self.contextual_bias_layers:
            bias = bias_layer(bias, attention_mask)
        scores += bias

        if attention_mask is not None:
            scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(out)


class DCBATransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, num_bias_layers, num_bias_heads, dropout=0.1,
                 layer_norm_eps=1e-5):
        super().__init__()
        self.attention = DCBAttention(hidden_size, num_heads, num_bias_layers, num_bias_heads, dropout, layer_norm_eps)
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
        self.num_bias_heads = cfg.num_bias_heads
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [DCBATransformerBlock(cfg.hidden_size, cfg.num_heads, cfg.hidden_size * 4,
                                  cfg.num_bias_layers, cfg.num_bias_heads,
                                  cfg.dropout, cfg.layer_norm_eps)
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
