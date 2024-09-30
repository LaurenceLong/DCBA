import math

from config import CustomConfig
from models.base_model import CustomTransformerBase, RotaryEmbedding, generate_casual_mask
from positional_encoding import generate_positions


class CustomTransformer(CustomTransformerBase):

    def __init__(self, cfg: CustomConfig):
        super().__init__(cfg=cfg)
        self.rotary_emb = RotaryEmbedding(cfg.hidden_size // cfg.num_heads)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        # Embedding
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.dropout(x)
        # Generate custom attention mask
        if attention_mask is None:
            attention_mask = generate_casual_mask(batch_size, self.num_heads, seq_len).to(x.device)
        # 添加位置编码
        batch_size, num_heads, seq_length, _ = attention_mask.shape
        # Generate position indices
        positions = generate_positions(seq_len, x.device)

        # Generate RoPE embeddings
        freqs_cos, freqs_sin = self.rotary_emb(positions)
        freqs_cos = freqs_cos.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        freqs_sin = freqs_sin.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # Process through transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
        # Output layer
        logits = self.fc(x)
        return logits
