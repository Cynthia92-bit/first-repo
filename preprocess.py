import torch
import json
import os
import spacy
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ================= 1. 分词器加载 =================
def load_tokenizers():
    def get_model(name):
        try:
            return spacy.load(name)
        except OSError:
            print(f"Downloading {name}...")
            os.system(f"python -m spacy download {name}")
            return spacy.load(name)
    return get_model("de_core_news_sm"), get_model("en_core_web_sm")

def tokenize(text, nlp):
    return [tok.text.lower() for tok in nlp.tokenizer(text)]

# ================= 2. 词汇表类 (完全替代 torchtext.vocab) =================
class Vocab:
    def __init__(self, counter, min_freq=2):
        # 保持与屏幕截图 2026-05-08 221605.png 一致的特殊符号
        self.itos = ["<blank>", "<s>", "</s>", "<unk>"]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        
        for tok, freq in counter.items():
            if freq >= min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
        
        self.blank_idx = self.stoi["<blank>"]
        self.bos_idx = self.stoi["<s>"]
        self.eos_idx = self.stoi["</s>"]
        self.unk_idx = self.stoi["<unk>"]

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.bos_idx] + [self.stoi.get(t, self.unk_idx) for t in tokens] + [self.eos_idx]

# ================= 3. 数据集与加载逻辑 =================
class Multi30kDataset(Dataset):
    def __init__(self, json_path, src_vocab, tgt_vocab, spacy_de, spacy_en):
        self.src_data = []
        self.tgt_data = []
        
        print(f"Loading {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # 分词并编码
                de_tokens = tokenize(item['de'], spacy_de)
                en_tokens = tokenize(item['en'], spacy_en)
                
                self.src_data.append(torch.tensor(src_vocab.encode(de_tokens)))
                self.tgt_data.append(torch.tensor(tgt_vocab.encode(en_tokens)))

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    # 使用 0 (<blank>) 进行填充
    src_padded = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_padded = pad_sequence(tgt_batch, padding_value=0, batch_first=True)
    return src_padded, tgt_padded

# ================= 4. 主运行逻辑 (对应你的 build_vocabulary) =================
def run_preprocessing():
    spacy_de, spacy_en = load_tokenizers()

    # 1. 构建词汇表 (仅使用训练集统计词频)
    print("Building Vocabularies...")
    counter_de = Counter()
    counter_en = Counter()
    
    with open("data/train.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            counter_de.update(tokenize(item['de'], spacy_de))
            counter_en.update(tokenize(item['en'], spacy_en))

    vocab_src = Vocab(counter_de, min_freq=2)
    vocab_tgt = Vocab(counter_en, min_freq=2)
    print(f"Vocab sizes: DE={len(vocab_src)}, EN={len(vocab_tgt)}")

    # 2. 创建 DataLoader
    train_dataset = Multi30kDataset("data/train.jsonl", vocab_src, vocab_tgt, spacy_de, spacy_en)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # 3. 验证数据读取
    for src, tgt in train_loader:
        print("Batch Source Shape:", src.shape) # 应该是 [32, max_len]
        print("Batch Target Shape:", tgt.shape)
        break
    
    return vocab_src, vocab_tgt, train_loader

if __name__ == "__main__":
    vocab_src, vocab_tgt, train_loader = run_preprocessing()
    #torch.save((vocab_src, vocab_tgt), "vocab.pt")