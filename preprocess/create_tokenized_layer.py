import json
from tqdm import tqdm 
import pickle
import argparse
import nltk 
import os 
def create_tokenized_layer(layer: dict = None, output_path:str = None, tokenized_text_path:str =None):
    
    new_layer = layer.copy()
    titles_path = os.path.join(tokenized_text_path, 'tokenized_raw_titles.txt')
    with open(titles_path,'rb') as f:
        tokenized_titles = pickle.load(f)
        
    ingrs_path = os.path.join(tokenized_text_path, 'tokenized_raw_ingrs.txt')
    with open(ingrs_path,'rb') as f:
        tokenized_ingrds = pickle.load(f)
        
    instrs_path = os.path.join(tokenized_text_path, 'tokenized_raw_instrs.txt')
    with open(instrs_path,'rb') as f:
        tokenized_instrs = pickle.load(f)
        
    for i, (k, v) in tqdm(enumerate(new_layer.items())):
        v['title'] = tokenized_titles[v['id']]
        v['ingredients'] = tokenized_ingrds[v['id']]
        v['instructions'] = tokenized_instrs[v['id']]

        
    with open(output_path,'wb') as f:
        pickle.dump(new_layer, f) 


def tokenize_and_save(vocab_path: str, entity: str ='title', layer=None, out_path: str = None):
    """ given a vocab, get and save the indices of each word in the dataset"""
    filter_ = ['-', '_', '/']

    embedded = {}
    count = 0
    with open(vocab_path,'rb') as f:
        vocab = pickle.load(f)
    for i, (k, v) in tqdm(enumerate(layer.items())):
        if entity == 'title':
            text = v['title'].lower()
            for f in filter_:
                text = text.replace(f, ' ')
            text = nltk.tokenize.word_tokenize(text)
            tokenized_text = []
            tokenized_text.append(vocab['<start>'])
            for w in text:
                if w in vocab:
                    tokenized_text.append(vocab[w])
                else:
                    tokenized_text.append(vocab['<unk>'])
                    count+=1
            tokenized_text.append(vocab['<end>'])
            embedded[v['id']] = tokenized_text
        elif entity == 'ingrs':
            tokenized_texts = []
            ingrs = [ing['text'].lower() for ing in v['ingredients']]
            for text in ingrs:
            
                for f in filter_:
                    text = text.replace(f, ' ')
                text = nltk.tokenize.word_tokenize(text)
                tokenized_text = []
                tokenized_text.append(vocab['<start>'])
                for w in text:
                    if w in vocab:
                        tokenized_text.append(vocab[w])
                    else:
                        tokenized_text.append(vocab['<unk>'])
                        count+=1
                tokenized_text.append(vocab['<end>'])
                tokenized_texts.append(tokenized_text)
            embedded[v['id']] = tokenized_texts
            
        elif entity == 'instrs':
            tokenized_texts = []
            ingrs = [ing['text'].lower() for ing in v['instructions']]
            for text in ingrs:
            
                for f in filter_:
                    text = text.replace(f, ' ')
                text = nltk.tokenize.word_tokenize(text)
                tokenized_text = []
                tokenized_text.append(vocab['<start>'])
                for w in text:
                    if w in vocab:
                        tokenized_text.append(vocab[w])
                    else:
                        tokenized_text.append(vocab['<unk>'])
                        count+=1
                        print(w)
                tokenized_text.append(vocab['<end>'])
                tokenized_texts.append(tokenized_text)
            embedded[v['id']] = tokenized_texts

    print(count, 'UNK')
    if out_path is not None:
        with open(out_path,'wb') as f:
            pickle.dump(embedded, f) 
    else:
        return embedded


if __name__ == '__main__':
	
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_layer', type=str,
                        help='path to layer 1')
    parser.add_argument('--output_path_vocab', type=str,)
    parser.add_argument('--output_path_layer1', type=str,)
    parser.add_argument('--output_path_tokenized_texts', type=str)

    args = parser.parse_args()



    path_layer1 = args.path_layer

    with open(path_layer1, 'r') as f:
        layer1 = json.load(f)
    layer1_ = {data['id']:data for data in tqdm(layer1)}

    print('finish reading')
    print('start creating vocab...')
    text = set()
    filter_titles = ['-', '_', '/']
    for i, (k, v) in tqdm(enumerate(layer1_.items())):
        title = v['title'].lower()
        ingrs = [ing['text'].lower() for ing in v['ingredients']]
        insts = [inst['text'].lower() for inst in v['instructions']]
        total = [title] + ingrs + insts
        for t in total:
            for f in filter_titles:
                t = t.replace(f, ' ')
            t = nltk.tokenize.word_tokenize(t)
            for w in t:
                text.add(w)

    text.add('<start>')
    text.add('<end>')
    text.add('<unk>')

    text_dict = {}
    for i, t in tqdm(enumerate(text)):
        text_dict[t] = i


    output_path_vocab = args.output_path_vocab

    with open(output_path_vocab,'wb') as f:
        pickle.dump(text_dict, f)

    print("start tokenization...")

    out_path = os.path.join(args.output_path_tokenized_texts, 'tokenized_raw_titles.txt')
    embedded = tokenize_and_save(vocab_path=output_path_vocab, entity='title', layer=layer1_, out_path=out_path)

    out_path = os.path.join(args.output_path_tokenized_texts, 'tokenized_raw_ingrs.txt') 
    embedded = tokenize_and_save(vocab_path=output_path_vocab, entity='ingrs', layer=layer1_, out_path=out_path)

    out_path = os.path.join(args.output_path_tokenized_texts, 'tokenized_raw_instrs.txt') 
    embedded = tokenize_and_save(vocab_path=output_path_vocab, entity='instrs', layer=layer1_, out_path=out_path)


    print('create and save layer 1...')
    create_tokenized_layer(layer=layer1_, output_path=args.output_path_layer1 , tokenized_text_path=args.output_path_tokenized_texts)