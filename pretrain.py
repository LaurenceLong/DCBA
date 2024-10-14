import logging
import os
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from config import CustomConfig, InitFrom
from data.prepare_text import CustomTextDataset
from evaluate import generate_text, decode_text
from model import DCBATransformer
from tokenizer import MixedTokenizer


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"training_{timestamp}.log"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = "cpu"


def train(model, train_dataloader, val_dataloader, config):
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=config.betas,
                            weight_decay=config.weight_decay)

    best_val_loss = float('inf')
    step = 0

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0

        with logging_redirect_tqdm():
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}", leave=True):
                input_ids, target_ids = batch
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)  # 将输入移动到设备

                optimizer.zero_grad()
                # 使用截短的input_ids
                char_logits = model(input_ids)
                loss = model.compute_loss(char_logits, target_ids)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                step += 1

                if step % config.log_interval == 0:
                    log.info(f"Step {step}, Loss: {loss.item():.4f}")

                if step % config.save_interval == 0:
                    torch.save(model.state_dict(), f'model_step_{step}.pth')
                    log.info(f"Saved model at step {step}")

                if step % config.eval_interval == 0:
                    val_loss = validate(model, val_dataloader, config)
                    log.info(f"Validation loss: {val_loss:.4f}")

                    prompt = decode_text(input_ids, tokenizer)
                    res_text = generate_text(model, tokenizer, prompt, max_new_tokens=10)
                    print(f"Generate for prompt: {prompt}")
                    print(res_text)
                    model.train()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), 'best_model.pth')
                        log.info("Saved best model.")

            if epoch == config.num_epochs - 1:
                val_loss = validate(model, val_dataloader, config)
                log.info(f"Validation loss: {val_loss:.4f}")

                prompt = decode_text(input_ids, tokenizer)
                res_text = generate_text(model, tokenizer, prompt, max_new_tokens=10)
                print(f"Generate for prompt: {prompt}")
                print(res_text)
                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), 'best_model.pth')
                    log.info("Saved best model.")

            avg_train_loss = total_loss / len(train_dataloader)
            log.info(f"Epoch {epoch + 1}, Average train loss: {avg_train_loss:.4f}")


def validate(model, dataloader, config):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= config.eval_iters:
                break
            input_ids, target_ids = batch
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)  # 将输入移动到设备

            logits = model(input_ids)
            loss = model.compute_loss(logits, target_ids)

            total_loss += loss.item()

    return total_loss / min(config.eval_iters, len(dataloader))


if __name__ == "__main__":
    log = get_logger(log_filename)
    # 配置
    config = CustomConfig()

    # 加载tokenizer
    tokenizer = MixedTokenizer()

    # 更新配置中的词汇表大小
    config.vocab_size = tokenizer.vocab_size
    config.char_vocab_size = tokenizer.u8_vocab_size
    config.token_vocab_size = tokenizer.token_vocab_size

    # 准备数据
    cwd = os.path.dirname(os.path.abspath(__file__))
    train_ds = [os.path.join(cwd, 'data/arithmetic_data.txt')]
    val_ds = [os.path.join(cwd, 'data/arithmetic_validation.txt')]
    train_dataset = CustomTextDataset(train_ds, tokenizer, config.max_seq_len)
    val_dataset = CustomTextDataset(val_ds, tokenizer, config.max_seq_len)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    # 初始化模型
    model = DCBATransformer(config)
    # 最后加载模型权重
    if config.init_from == InitFrom.resume:
        model.load_state_dict(torch.load('best_model.pth'))

    # 训练模型
    train(model, train_dataloader, val_dataloader, config)
