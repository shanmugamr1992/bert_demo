import os 
import tqdm
import sys

from pathlib import Path
sys.path.append("./src")

from bert_model.bert_model import BertModel
from dataset.bert_dataset import BERTDataset
from torch.utils.data import DataLoader
from trainer.bert_trainer import BERTTrainer
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer



MAX_LEN = 64

def get_data_pairs():
    ### loading all data into memory
    corpus_movie_conv = './datasets/movie_conversations.txt'
    corpus_movie_lines = './datasets/movie_lines.txt'
    with open(corpus_movie_conv, 'r', encoding='iso-8859-1') as c:
        conv = c.readlines() 
    with open(corpus_movie_lines, 'r', encoding='iso-8859-1') as l:
        lines = l.readlines()

    lines_dic = {}
    for line in lines:
        objects = line.split(" +++$+++ ")
        lines_dic[objects[0]] = objects[-1]

    pairs = []
    for con in conv:
        ids = eval(con.split(" +++$+++ ")[-1])
        for i in range(len(ids)):
            qa_pairs = []
            
            if i == len(ids) - 1:
                break

            first = lines_dic[ids[i]].strip()  
            second = lines_dic[ids[i+1]].strip() 

            qa_pairs.append(' '.join(first.split()[:MAX_LEN]))
            qa_pairs.append(' '.join(second.split()[:MAX_LEN]))
            pairs.append(qa_pairs)
    return pairs

def save_tokenizer():

    os.mkdir('./data')
    text_data = []
    file_count = 0
    pairs = get_data_pairs()
    for sample in tqdm.tqdm([x[0] for x in pairs]):
        text_data.append(sample)

        # once we hit the 10K mark, save to file
        if len(text_data) == 10000:
            with open(f'./data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(text_data))
            text_data = []
            file_count += 1

    paths = [str(x) for x in Path('./data').glob('**/*.txt')]    
    ### training own tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True
    )

    tokenizer.train( 
        files=paths,
        vocab_size=30_000, 
        min_frequency=5,
        limit_alphabet=1000, 
        wordpieces_prefix='##',
        special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
        )

    os.mkdir('./bert-it-1')
    tokenizer.save_model('./bert-it-1', 'bert-it')    

def get_tokenizer():
    return BertTokenizer.from_pretrained('./src/tokenizer/bert-it-vocab.txt', local_files_only=True)

if __name__ == "__main__":

    pairs = get_data_pairs()

    tokenizer = get_tokenizer()

    train_data = BERTDataset(pairs, seq_len=MAX_LEN, tokenizer=tokenizer)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)

    bert_model =  BertModel(tokenizer.vocab_size)

    bert_trainer = BERTTrainer(bert_model, train_loader, device='cpu')
    epochs = 4

    for epoch in range(epochs):
        bert_trainer.train(epoch)

# docker run -it -v /home/shanmugamr/check_bert/data:/workspace/data -v /home/shanmugamr/bert_demo:/workspace/bert_demo nvcr.io/nvidia/pytorch:23.08-py3  bash
# pip install tokenizers transformers
# python3 pretrain_bert.py