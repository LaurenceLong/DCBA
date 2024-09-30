# coding=utf-8
# DEFAULT_VOCAB_SIZE = 983040  # 983040 是 U+F0000 的十进制值
DEFAULT_VOCAB_SIZE = 250  # 只用一个字节


class UTF8Tokenizer:
    def __init__(self, vocab_size=DEFAULT_VOCAB_SIZE):
        self.vocab_size = min(vocab_size, DEFAULT_VOCAB_SIZE)
        special_tokens = {'<BOS>': 1, '<EOS>': 2, '<UNK>': 3, '<BOB>': 4, '<EOB>': 5}
        self.special_tokens = {'<PAD>': 0}
        self.special_tokens.update({k: v + self.vocab_size for k, v in special_tokens.items()})
        self.char_to_id = {chr(i): i + len(self.special_tokens) for i in
                           range(min(128, vocab_size - len(self.special_tokens)))}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

        for token, _id in self.special_tokens.items():
            self.id_to_char[_id] = token

        self.pad_token_id = self.special_tokens['<PAD>']
        self.bos_token_id = self.special_tokens['<BOS>']
        self.eos_token_id = self.special_tokens['<EOS>']
        self.unk_token_id = self.special_tokens['<UNK>']
        self.bob_token_id = self.special_tokens['<BOB>']
        self.eob_token_id = self.special_tokens['<EOB>']

    def encode(self, text, add_special_tokens=False):
        encoded = []
        if add_special_tokens:
            encoded.append(self.bos_token_id)
        for char in text:
            if char in self.char_to_id:
                encoded.append(self.char_to_id[char])
            else:
                # 处理UTF-8字符
                for byte in char.encode('utf-8'):
                    if byte < self.vocab_size:
                        encoded.append(byte)
                    else:
                        encoded.append(self.unk_token_id)
        if add_special_tokens:
            encoded.append(self.eos_token_id)
        return encoded

    def decode(self, tokens):
        text = []
        utf8_bytes = []
        for token in tokens:
            if token in self.special_tokens.values():
                continue
            if token in self.id_to_char:
                if utf8_bytes:
                    text.append(bytes(utf8_bytes).decode('utf-8', errors='replace'))
                    utf8_bytes = []
                text.append(self.id_to_char[token])
            else:
                utf8_bytes.append(token)
        if utf8_bytes:
            text.append(bytes(utf8_bytes).decode('utf-8', errors='replace'))
        return ''.join(text)

    def __len__(self):
        return self.vocab_size

    def get_vocab(self):
        vocab = {**self.special_tokens, **self.char_to_id}
        return {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}

    def convert_ids_to_tokens(self, ids):
        return [self.id_to_char.get(_id, chr(_id)) if _id not in self.special_tokens.values() else self.id_to_char[_id]
                for _id in ids]

    def convert_tokens_to_ids(self, tokens):
        return [
            self.char_to_id.get(token, ord(token)) if token not in self.special_tokens else self.special_tokens[token]
            for token in tokens]

    def save_pretrained(self, path):
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab_size': self.vocab_size,
                'special_tokens': self.special_tokens,
                'char_to_id': self.char_to_id
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, path):
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tokenizer = cls(data['vocab_size'])
        tokenizer.special_tokens = data['special_tokens']
        tokenizer.char_to_id = data['char_to_id']
        tokenizer.id_to_char = {v: k for k, v in tokenizer.char_to_id.items()}
        for token, _id in tokenizer.special_tokens.items():
            tokenizer.id_to_char[_id] = token
        return tokenizer

