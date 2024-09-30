import numpy as np
import torch
import math

from models.base_model import generate_casual_mask


def get_slopes(n_heads):
    def get_slopes_power_of_2(n_heads):
        start = (2 ** (-2 ** -(math.log2(n_heads) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n_heads)]

    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                           :n_heads - closest_power_of_2]


def generate_positions(seq_length, device):
    return torch.arange(seq_length, dtype=torch.float32, device=device).unsqueeze(0)


def generate_rev_distance_matrix(seq_length, device):
    distances = generate_positions(seq_length, device)
    return distances.unsqueeze(1) - distances.unsqueeze(2)


def build_alibi_tensor(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    batch_size, num_heads, seq_length, _ = attention_mask.shape
    # 生成距离矩阵
    distances = generate_rev_distance_matrix(seq_length, device=attention_mask.device)
    # 获取每个头的斜率
    slopes = torch.tensor(get_slopes(num_heads), dtype=dtype, device=attention_mask.device)
    # 计算alibi偏置
    alibi = slopes.view(1, num_heads, 1, 1) * distances.unsqueeze(1)
    return alibi


def test_build_alibi_tensor():
    # 测试用例1：基本功能测试
    batch_size = 1
    num_heads = 1
    seq_len = 9
    pos = 3

    input_ids = torch.ones(batch_size, seq_len)
    device = "cpu"

    print(1111, get_slopes(num_heads))

    print(2222, generate_positions(seq_len, device))

    print(3333, generate_rev_distance_matrix(seq_len, device))

    attn_mask = generate_casual_mask(batch_size, num_heads, seq_len)
    print(4444, build_alibi_tensor(attn_mask, dtype=torch.float32))


if __name__ == "__main__":
    # 运行测试
    test_build_alibi_tensor()
