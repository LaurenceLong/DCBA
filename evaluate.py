# 加载最佳模型
import os

import torch

from config import CustomConfig
from model import DCBATransformer
from tokenizer import MixedTokenizer


def show_model_parameters(model):
    print(f"Model params:")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Elements: {param.numel()}")


def decode_text(generated, tokenizer):
    return tokenizer.decode(generated[0])


def generate_text(model, tokenizer, prompt, max_new_tokens):
    current = prompt
    inited = False
    for _ in range(max_new_tokens):
        input_ids = torch.tensor(tokenizer.encode(current)).to(next(model.parameters()).device)
        # print(3333, input_ids, start_pos)
        if not inited:
            inited = True
        # generated = model.generate(input_ids, max_new_tokens, tokenizer.eos_token_id, tokenizer=tokenizer)
        generated = model.generate(input_ids, max_new_tokens, tokenizer.eos_token_id)
        # print(4444, generated)
        current = decode_text(generated, tokenizer)
        if generated[0][-1] == tokenizer.eos_token_id:
            break
    return current


def valid_generate(model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    tokenizer = MixedTokenizer()

    config = CustomConfig()
    config.vocab_size = tokenizer.vocab_size
    config.dropout = 0
    model = DCBATransformer(config)
    # 最后加载模型权重
    if model_path is None:
        model.load_state_dict(torch.load('best_model.pth'))
    else:
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    show_model_parameters(model)
    # 生成文本示例
    while True:
        prompt = input("Input your prompt: ")
        if len(prompt) == 0:
            continue
        res_text = generate_text(model, tokenizer, prompt, max_new_tokens=10)
        print(f"Generated text: {res_text}")


def batch_evaluate(file_name="data/arithmetic_test.txt", model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    tokenizer = MixedTokenizer()
    config = CustomConfig()
    config.vocab_size = tokenizer.vocab_size
    config.dropout = 0
    model = DCBATransformer(config)
    # 最后加载模型权重
    if model_path is None:
        model.load_state_dict(torch.load('best_model.pth'))
    else:
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    show_model_parameters(model)

    root = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(root, file_name)
    with open(file_path, encoding="utf-8") as fd:
        lines = fd.readlines()
    for li in lines:
        if "=" in li:
            idx = li.find("=")
            prompt = li[:idx + 1] + " "
            result = li[idx + 1:].strip()
            res_text = generate_text(model, tokenizer, prompt, 10)
            print(f"Calculate: {prompt}? ({result})")
            print(f"Generated: {res_text}")
            print(f"=" * 50)


if __name__ == "__main__":
    pth = None
    # pth = '/home/laurence/work/ai/DCBATransformer/expirements/alibi/best_model.pth'
    # valid_generate(pth)
    # pth = '/home/laurence/work/ai/DCBATransformer/expirements/rope/best_model.pth'
    batch_evaluate(model_path=pth)
