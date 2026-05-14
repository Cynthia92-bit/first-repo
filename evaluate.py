import torch
import os
import sacrebleu
import json
from transformer import make_model, greedy_decode
from preprocess import load_tokenizers, tokenize

def run_evaluation():
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path="checkpoints/model_epoch_40.pt"
    vocab_path="vocab.pt"
    test_data_path = "data/test.jsonl"
    print(f"正在使用设备 :{device}")

    try:
       data = torch.load(vocab_path, map_location=device, weights_only=False) # 尝试加载
       vocab_src, vocab_tgt = data
    except:
        print("发生错误")
        return
    spacy_de, spacy_en = load_tokenizers()
    model=make_model(len(vocab_src),len(vocab_tgt),N=6)

    model.load_state_dict(torch.load(model_path,map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    test_samples=[]
    with open(test_data_path,'r',encoding='utf-8') as f:
        for line in f:
            test_samples.append(json.loads(line))
    hypothesis=[]
    references = []
    print(f"开始推理{len(test_data_path)}")

    with torch.no_grad():
        for i,sample in enumerate(test_samples):
            src_text=sample['de']
            tgt_text = sample['en']
            tokens=["<s>"]+tokenize(src_text,spacy_de)+["</s>"]
            src_indices=[vocab_src.stoi.get(t,vocab_src.stoi["<unk>"]) for t in tokens]
            src_tensor=torch.LongTensor([src_indices]).to(device)
            src_mask=(src_tensor!=vocab_src.blank_idx).unsqueeze(-2)  #布尔矩阵

            start_id=vocab_tgt.stoi.get("<s>", 0)
            out=greedy_decode(model,src_tensor,src_mask,max_len=72,start_symbol=start_id)
            pred_words=[]
            for j in range(1,out.size(1)):
                sym=vocab_tgt.itos[out[0,j].item()]
                if sym =="</s>":
                    break
                pred_words.append(sym)   #将索引转回文字
            pred_sentence=" ".join(pred_words)
            hypothesis.append(pred_sentence)
            references.append(tgt_text)   #准备进入bleu
            if (i+1)%100==0:
                print(f"进度：{i+1}/{len(test_samples)}")
                print(f"源语：{src_text}")
                print(f"预测：{pred_sentence}")
                print(f"目标：{tgt_text}\n")
    bleu=sacrebleu.corpus_bleu(hypothesis,[references], tokenize='13a')
    print("\n" + "="*30)
    print(f"测试完成！")
    print(f"BLEU 分数: {bleu.score:.2f}")
    print("="*30)
if __name__=="__main__":
    run_evaluation()