from dataclasses import dataclass


class InitFrom:
    scratch: int = 0
    resume: int = 1


@dataclass
class CustomConfig:
    # model params
    hidden_size: int = 128
    num_heads: int = 4
    num_layers: int = 16
    vocab_size: int = -1
    max_seq_len: int = 128
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    # training params...
    learning_rate: float = 3e-4
    weight_decay: float = 1e-1
    betas: tuple = (0.9, 0.95)
    batch_size: int = 16
    num_epochs: int = 40
    log_interval: int = 20
    eval_interval: int = 20
    eval_iters: int = 10
    save_interval: int = 5000
    init_from: int = InitFrom.scratch
