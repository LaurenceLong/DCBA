import os

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.nested_list_index import NestedListIndex
from tokenizer import MixedTokenizer

CWD = os.path.dirname(os.path.abspath(__file__))


class CustomTextDataset(Dataset):
    def __init__(self, data_paths, tokenizer, max_seq_len=256, cache_dir=os.path.join(CWD, '.cache')):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        self.data = []
        self.data_lengths = []
        self.data_idx = []

        for path in data_paths:
            cache_file = os.path.join(cache_dir, f'{os.path.basename(path)}.parquet')
            if not os.path.exists(cache_file):
                os.makedirs(cache_dir, exist_ok=True)
                self.process_file(path, tokenizer, cache_file)

            table = pq.read_table(cache_file)
            self.data.append(table)
            self.data_lengths.append(len(table.column('lengths').to_pylist()))

        self.total_length = sum(self.data_lengths)
        self.nested_list_index = NestedListIndex(self.data_lengths)

    @staticmethod
    def process_file(input_file, tokenizer: MixedTokenizer, output_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        tokenized_lines = []
        lengths = []
        for line in lines:
            tokens = tokenizer.encode(line.strip().strip('\n').strip('\r'), add_special_tokens=True)[1:]
            tokenized_lines.append(tokens)
            lengths.append(len(tokens))

        table = pa.Table.from_arrays(
            [pa.array(tokenized_lines, type=pa.list_(pa.uint16())), lengths],
            names=['tokens', 'lengths']
        )
        pq.write_table(table, output_file)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        data_index, elem_idx = self.nested_list_index.find_list_index(idx)
        sample = self.data[data_index].column('tokens')[elem_idx].as_py()

        # 使用切片操作来限制样本长度
        sample = sample[-self.max_seq_len - 1:]  # 多取一个元素，为了后面的src和tgt

        # 使用torch.tensor()一次性创建张量，避免多次转换
        tensor = torch.tensor(sample, dtype=torch.long)

        src = tensor[:-1]  # 所有元素除了最后一个
        tgt = tensor[1:]  # 所有元素除了第一个

        # 使用 F.pad 进行填充，这通常比手动填充更快
        if len(src) < self.max_seq_len:
            src = F.pad(src, (0, self.max_seq_len - len(src)), value=0)
            tgt = F.pad(tgt, (0, self.max_seq_len - len(tgt)), value=0)

        return src, tgt

