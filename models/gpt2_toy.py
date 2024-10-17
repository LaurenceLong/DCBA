import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2Config, AdamW, get_scheduler

from data.prepare_text import CustomTextDatasetFlat
from tokenizer import MixedTokenizer


def pretrain(config, train_dataset, num_epochs=100, batch_size=16, learning_rate=5e-5, save_path="./gpt2_toy_model"):
    # 使用这个配置创建一个新的GPT2LMHeadModel实例
    model = GPT2LMHeadModel(config)

    # 准备数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 设置学习率调度器
    num_training_steps = len(train_dataloader) * num_epochs
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # 将模型移动到GPU（如果有）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 训练模型
    model.train()
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            inputs = batch.to(device)

            # 获取损失
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss

            # 反向传播
            loss.backward()

            # 优化器更新
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # 保存微调后的模型
    model.save_pretrained(save_path)


def evaluate(model_path, tokenizer, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 从保存的权重加载模型
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    # 生成文本示例
    while True:
        prompt = input("Input your prompt: ")
        if len(prompt) == 0:
            continue
        # 使用tokenizer将prompt转换为list[int]
        input_ids = tokenizer.encode(prompt)

        # 将list[int]转换为tensor，并添加批次维度
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

        # 创建attention_mask
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=20,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )

        # 将生成的token ID转换回文本
        generated_text = tokenizer.decode(outputs[0].tolist())
        print(generated_text)


if __name__ == "__main__":
    # 加载分词器
    tokenizer = MixedTokenizer()

    # 创建一个新的GPT2Config，使用与"gpt2"相同的配置
    config = GPT2Config.from_pretrained("gpt2")

    # 由于GPT-2没有pad token，我们用eos_token来填充
    tokenizer.pad_token = tokenizer.eos_token_id
    config.vocab_size = tokenizer.vocab_size

    # 使用Hugging Face的`datasets`库加载数据集
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_ds = [os.path.join(root, 'data/arithmetic_data.txt')]
    train_dataset = CustomTextDatasetFlat(train_ds, tokenizer, 128)

    # 预训练模型
    # pretrain(config, train_dataset)

    # 评估模型
    evaluate("./gpt2_toy_model", tokenizer)
