from utf8_tokenizer import UTF8Tokenizer


class MixedTokenizer:
    def __init__(self):
        # import tiktoken
        # self.gpt4o_tokenizer = tiktoken.get_encoding("o200k_base")  # GPT4o
        # self.vocab_size = self.u8_vocab_size + self.gpt4o_tokenizer.n_vocab
        from transformers import GPT2TokenizerFast
        self.gpt4o_tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/gpt-4o')
        self.token_vocab_size = len(self.gpt4o_tokenizer)
        special_tokens = {'<BOS>': 1, '<EOS>': 2, '<UNK>': 3, '<BOB>': 4, '<EOB>': 5}
        self.special_tokens = {'<PAD>': 0}
        self.special_tokens.update({k: v + self.token_vocab_size for k, v in special_tokens.items()})
        self.vocab_size = self.token_vocab_size + len(self.special_tokens)

        self.pad_token_id = self.special_tokens['<PAD>']
        self.bos_token_id = self.special_tokens['<BOS>']
        self.eos_token_id = self.special_tokens['<EOS>']
        self.unk_token_id = self.special_tokens['<UNK>']
        self.bob_token_id = self.special_tokens['<BOB>']
        self.eob_token_id = self.special_tokens['<EOB>']

    def encode(self, text, add_special_tokens=False):
        raw = self.gpt4o_tokenizer.encode(text)
        encoded = []
        if add_special_tokens:
            encoded.append(self.bos_token_id)
        for r in raw:
            encoded.append(r)
        if add_special_tokens:
            encoded.append(self.eos_token_id)
        return encoded

    def decode(self, tokens):
        filtered_tokens = []
        for token in tokens:
            if token in self.special_tokens.values():
                continue
            filtered_tokens.append(token)
        return self.gpt4o_tokenizer.decode(filtered_tokens)


if __name__ == "__main__":
    mixed_tokenizer = MixedTokenizer()
    b = mixed_tokenizer.encode("Hello world")
    print(b)
    d = mixed_tokenizer.decode(b)
    print(d)
