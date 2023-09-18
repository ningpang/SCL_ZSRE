# -*- encoding: utf-8 -*-

"""snli数据预处理"""
import argparse
import os
import time
import json
from tqdm import tqdm
import numpy as np
import random


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

        if cur_pos == pos_head[-1]:
            tokens.append('</e1>')
        if cur_pos == pos_tail[-1]:
            tokens.append('</e2>')
        cur_pos += 1

    return ' '.join(tokens), first_pos


@timer
def fewrel_preprocess(train_src_path, train_dst_path, val_src_path, val_dst_path, way, kr, ks):
    """处理原始的中文snli数据

    Args:
        src_path (str): 原始文件地址
        dst_path (str): 输出文件地址
    """
    pid2name = json.load(open('./FewRel/pid2name.json'))
    # 组织数据
    with open(train_src_path, 'r', encoding='utf-8') as reader,\
        open(train_dst_path + '/train.txt', 'w', encoding='utf-8') as writer, \
        open(val_src_path, 'r', encoding='utf-8') as reader1, \
        open(val_dst_path + '/val.txt', 'w', encoding='utf-8') as writer1:

        json_data = json.load(reader)
        json_data = dict(json.load(reader1), **json_data)

        all_keys = list(json_data.keys())

        test_keys = np.random.choice(all_keys, args.way, replace=False)
        train_keys = set(json_data.keys()) - set(test_keys)
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
                    sent1, sent1_pos = cvt_data(item1['tokens'], item1['h'][2][0], item1['t'][2][0])
                    item2 = np.random.choice(raw_train[json_key], 1, replace=False)[0]
                    sent_e1, sent_e1_pos = cvt_data(item2['tokens'], item2['h'][2][0], item2['t'][2][0])
                    sent_e2 = ','.join(pid2name[json_key])
                    cont_keys = np.random.choice(list(set(train_keys) - set([json_key])), kr, replace=False)
                    for cont_key in cont_keys:
                        item1 = np.random.choice(raw_train[cont_key], 1, replace=False)[0]
                        snet_c1, snet_c1_pos = cvt_data(item1['tokens'], item1['h'][2][0], item1['t'][2][0])
                        writer.write(json.dumps(
                            {'relation': json_key, 'origin': sent1, 'entailment': sent_e1, 'description': sent_e2,
                             'contradiction': snet_c1, 'pos': sent1_pos + sent_e1_pos + snet_c1_pos}) + '\n')

            else:
                length = int(args.ks)
                items = np.random.choice(raw_train[json_key], length, replace=False)
                for item1 in items:
                    sent1, sent1_pos = cvt_data(item1['tokens'], item1['h'][2][0], item1['t'][2][0])
                    item2 = np.random.choice(raw_train[json_key], 1, replace=False)[0]
                    sent_e1, sent_e1_pos = cvt_data(item2['tokens'], item2['h'][2][0], item2['t'][2][0])
                    sent_e2 = ','.join(pid2name[json_key])
                    cont_keys = np.random.choice(list(set(train_keys) - set([json_key])), kr, replace=False)
                    for cont_key in cont_keys:
                        item1 = np.random.choice(raw_train[cont_key], 1, replace=False)[0]
                        snet_c1, snet_c1_pos = cvt_data(item1['tokens'], item1['h'][2][0], item1['t'][2][0])
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
                sent, _ = cvt_data(instance['tokens'], instance['h'][2][0], instance['t'][2][0])
                instance['tokens'] = sent
                new_test[key].append(instance)

        writer1.write(json.dumps(new_test))


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


# 500 1 94
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 这个数据集有80个关系，实验设置15-20-25-30-40
    parser.add_argument('--train_way', type=int, default=40)
    parser.add_argument('--way', type=int, default=25)

    parser.add_argument('--seed', type=int, default=2023)

    parser.add_argument('--kr', type=int, default=1)
    parser.add_argument('--ks', type=str, default='200')
    args = parser.parse_args()
    for w in [20, 30, 35, 40]:
        args.way = w
        set_seed(args.seed)
        print(args)
        train_src, train_dst, val_src, val_dst = './FewRel/train_wiki.json', f'./FewRel/{args.way}-way-{args.ks}-shot/', './FewRel/val_wiki.json', f'./FewRel/{args.way}-way-{args.ks}-shot/'
        if not os.path.exists(train_dst):
            os.makedirs(train_dst)
        if not os.path.exists(val_dst):
            os.makedirs(val_dst)

        fewrel_preprocess(train_src, train_dst, val_src, val_dst, args.way, args.kr, args.ks)
