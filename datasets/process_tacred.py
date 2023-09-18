# -*- encoding: utf-8 -*-

"""snli数据预处理"""
import argparse
import os
import time
import json
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', never_split=['<e1>', '</e1>', '<e2>', '</e2>'])
special_tokens_dict = {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>']}  # add special token
tokenizer.add_special_tokens(special_tokens_dict)

def timer(func):
    """ time-consuming decorator
    """

    def wrapper(*args, **kwargs):
        ts = time.time()
        res = func(*args, **kwargs)
        te = time.time()
        print(f"function: `{func.__name__}` running time: {te - ts:.4f} secs")
        return res

    return wrapper


def cvt_data(raw_tokens, pos_head, pos_tail):
    # token -> index
    # tokens = ['[CLS]']
    tokens = []
    first_pos = []
    cur_pos = 0
    for token in raw_tokens:
        token = token.lower()
        if cur_pos == pos_head[0]:
            tokens.append('<e1>')
            first_pos.append(cur_pos)
        if cur_pos == pos_tail[0]:
            tokens.append('<e2>')
            first_pos.append(cur_pos)

        tokens.append(token)

        if cur_pos == pos_head[-1]-1:
            tokens.append('</e1>')
        if cur_pos == pos_tail[-1]-1:
            tokens.append('</e2>')
        cur_pos += 1

    return ' '.join(tokens), first_pos


@timer
def fewrel_preprocess(train_src_path, train_dst_path, val_src_path, val_dst_path, test_src_path, kr):
    """处理原始的中文snli数据

    Args:
        src_path (str): 原始文件地址
        dst_path (str): 输出文件地址
    """

    pid2name = json.load(open('./TACRED/pid2name.json'))
    # 组织数据
    with open(train_src_path, 'r', encoding='utf-8') as reader, \
        open(train_dst_path + '/train.txt', 'w', encoding='utf-8') as writer, \
        open(val_src_path, 'r', encoding='utf-8') as reader1, \
        open(val_dst_path + '/val.txt', 'w', encoding='utf-8') as writer1, \
        open(test_src_path, 'r', encoding='utf-8') as reader2:

        json_data = {}
        lines = []
        lines += reader.readlines()
        lines += reader1.readlines()
        lines += reader2.readlines()

        for line in lines:
            temp = json.loads(line)
            pid = temp['relation']
            if pid not in json_data:
                json_data[pid] = []
            item = {}
            item['tokens'] = temp['token']
            if len(temp['h']) == 0 or len(temp['t']) == 0:
                continue
            assert len(temp['h']) != 0
            assert len(temp['t']) != 0
            item['h'] = ['', '', [temp['h']]]
            item['t'] = ['', '', [temp['t']]]
            json_data[pid].append(item)

        few_keys = []
        many_keys = []

        for rel in json_data:
            if rel == 'NA':
                continue
            length = len(json_data[rel])
            if length > 200:
                many_keys.append(rel)
            else:
                few_keys.append(rel)

        new_json_data = {}
        for rel in json_data:
            if len(json_data[rel])>1000:
                new_json_data[rel] = json_data[rel][:1000]
            else:
                new_json_data[rel] = json_data[rel]
        json_data = new_json_data
        if len(few_keys) >= args.way:
            test_keys = np.random.choice(few_keys, args.way, replace=False)
            keys_else = set(few_keys) - set(test_keys)
            train_keys = many_keys + list(keys_else)
        else:
            add_num = args.way - len(few_keys)
            add_keys = np.random.choice(many_keys, add_num, replace=False)
            test_keys = few_keys + list(add_keys)
            train_keys = list(set(many_keys) - set(add_keys))

        train_keys_2_id = {name: idx for idx, name in enumerate(train_keys)}
        with open(train_dst_path + '/trainrel2id.json', 'w') as f:
            json.dump(train_keys_2_id, f)

        train_values = [json_data[k] for k in train_keys]
        raw_train = dict(zip(train_keys, train_values))
        test_values = [json_data[k] for k in test_keys]
        raw_test = dict(zip(test_keys, test_values))

        for json_key in train_keys:
            if args.ks == 'full':
                for item1 in raw_train[json_key]:
                    sent1, sent1_pos = cvt_data(item1['tokens'], item1['h'][2][0]['pos'], item1['t'][2][0]['pos'])
                    item2 = np.random.choice(raw_train[json_key], 1, replace=False)[0]
                    sent_e1, sent_e1_pos = cvt_data(item2['tokens'], item2['h'][2][0]['pos'], item2['t'][2][0]['pos'])
                    sent_e2 = ','.join(pid2name[json_key])
                    cont_keys = np.random.choice(list(set(train_keys) - set([json_key])), kr, replace=False)
                    for cont_key in cont_keys:
                        item1 = np.random.choice(raw_train[cont_key], 1, replace=False)[0]
                        snet_c1, snet_c1_pos = cvt_data(item1['tokens'], item1['h'][2][0]['pos'], item1['t'][2][0]['pos'])
                        writer.write(json.dumps(
                            {'relation': json_key, 'origin': sent1, 'entailment': sent_e1, 'description': sent_e2,
                             'contradiction': snet_c1, 'pos': sent1_pos + sent_e1_pos + snet_c1_pos}) + '\n')
            else:
                length = int(args.ks)
                if length <= len(raw_train[json_key]):
                    items = np.random.choice(raw_train[json_key], length, replace=False)
                else:
                    items = raw_train[json_key]
                for item1 in items:
                    sent1, sent1_pos = cvt_data(item1['tokens'], item1['h'][2][0]['pos'], item1['t'][2][0]['pos'])
                    item2 = np.random.choice(raw_train[json_key], 1, replace=False)[0]
                    sent_e1, sent_e1_pos = cvt_data(item2['tokens'], item2['h'][2][0]['pos'], item2['t'][2][0]['pos'])
                    sent_e2 = ','.join(pid2name[json_key])
                    cont_keys = np.random.choice(list(set(train_keys) - set([json_key])), kr, replace=False)
                    for cont_key in cont_keys:
                        item1 = np.random.choice(raw_train[cont_key], 1, replace=False)[0]
                        snet_c1, snet_c1_pos = cvt_data(item1['tokens'], item1['h'][2][0]['pos'], item1['t'][2][0]['pos'])
                        writer.write(json.dumps(
                            {'relation': json_key, 'origin': sent1, 'entailment': sent_e1, 'description': sent_e2,
                             'contradiction': snet_c1, 'pos': sent1_pos + sent_e1_pos + snet_c1_pos}) + '\n')
        new_test = {}
        for key in raw_test:
            if key in raw_train:
                print('ERROR!!!!!!!!!!!!!!!!')
            if key not in new_test:
                new_test[key] = []
            for instance in raw_test[key]:
                sent, _ = cvt_data(instance['tokens'], instance['h'][2][0]['pos'], instance['t'][2][0]['pos'])
                sent_feature = tokenizer.encode(sent)
                if len(sent_feature) > 200:
                    continue
                instance['tokens'] = sent
                new_test[key].append(instance)
        writer1.write(json.dumps(new_test))


def set_seed(seed):
    np.random.seed(seed)

# 500 1 94
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 这个数据集有80个关系，实验设置10-15-20-2
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2024)

    parser.add_argument('--kr', type=int, default=1)
    parser.add_argument('--ks', type=str, default='200')
    args = parser.parse_args()


    for w in [5, 10, 15, 20]:
        args.way = w
        set_seed(args.seed)
        print(args)
        train_src, train_dst, val_src, val_dst, test_src = './TACRED/train.txt', f'./TACRED/{args.way}-way-{args.ks}-shot/', './TACRED/val.txt', f'./TACRED/{args.way}-way-{args.ks}-shot/', './TACRED/test.txt'
        if not os.path.exists(train_dst):
            os.makedirs(train_dst)
        if not os.path.exists(val_dst):
            os.makedirs(val_dst)
        fewrel_preprocess(train_src, train_dst, val_src, val_dst, test_src, args.kr)
