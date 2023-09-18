import argparse
import math
import numpy as np
import time
import datetime
import os
import csv
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoConfig

from utils import set_seed, format_time, get_contrastive_data, get_contrastive_feature, get_entity_idx
from dataset import RCLdataset, collate_fn
from bert_encoder import Bert_Encoder
from model import PromptCL
from evaluation import ClusterEvaluation, standard_metric_result
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score, \
    v_measure_score

import warnings

warnings.filterwarnings('ignore')


def train(contrastive_dataset, model, moment, device, train_batch_size, train_epochs, seeds, save_model, collate_fn):
    # train_sampler = RandomSampler(contrastive_dataset)
    train_dataloader = DataLoader(contrastive_dataset, batch_size=train_batch_size,
                                  collate_fn=collate_fn, shuffle=True)

    t_total = len(train_dataloader) * train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.1},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
    # Train
    model.to(device)
    model.zero_grad()
    set_seed(seeds)

    training_stats = []
    global_step = 0
    best_eval_loss = 10
    # 统计整个训练时长
    # moment.init_moment(args, model, contrastive_dataset)

    for i in range(train_epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(i + 1, train_epochs))
        print('Training...')
        total_train_loss = 0
        t0 = time.time()

        for step, (data, ind) in enumerate(train_dataloader):
            model.train()

            if step % 500 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            # entity_idx = data['entity_idx'].to(device)
            if 'token_type_ids' in data.keys():
                token_type_ids = data['token_type_ids'].to(device)
            else:
                token_type_ids = None

            if 'label' in data.keys():
                classify_labels = data['label'].to(device)
            else:
                classify_labels = None

            loss = model(moment,
                         input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids,
                         classify_labels=classify_labels,
                         ind=ind,
                         # entity_idx=entity_idx,
                         use_cls=False
                         )  # mark

            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scheduler.step()
            model.zero_grad()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        print('Saving Model...')
        torch.save(model, save_model)

        test(i, save_model)

def test(i, model_path):
    # test
    t0 = time.time()
    model = torch.load(model_path)
    model.eval()
    sent_embs = []
    templates = ["In the following sentence, the relationship between ##entity_1## and ##entity_2## is [MASK].",
                 "The relationship between ##entity_1## and ##entity_2## is [MASK].",
                 "I think ##entity_1## is [MASK] of ##entity_2##.",
                 "In the following sentence, ##entity_1## is [MASK] of ##entity_2##.",
                 "##entity_1##? [MASK], ##entity_2##",
                 "I think [E11] ##entity_1## [E12] is [MASK] of [E21] ##entity_2## [E12].",
                 "[P00] The relationship between ##entity_1## and ##entity_2## is [MASK].",
                 "[P00] [E11] [E12] [E21] [E22] ##entity_1##? [MASK], ##entity_2##.",
                 ]
    with torch.no_grad():
        # e1_tks_id = tokenizer.convert_tokens_to_ids('<e1>')
        # e2_tks_id = tokenizer.convert_tokens_to_ids('<e2>')
        for unseen_sentence in test_sents:
            unseen_sentence = unseen_sentence.split(' ')
            p11 = unseen_sentence.index("<e1>") + 1
            p12 = unseen_sentence.index("</e1>")
            p21 = unseen_sentence.index("<e2>") + 1
            p22 = unseen_sentence.index("</e2>")
            entity_1 = " ".join(unseen_sentence[p11:p12])
            entity_2 = " ".join(unseen_sentence[p21:p22])
            sample_sentence = ' '.join(unseen_sentence).replace("<e1>", "").replace("</e1>", "").replace("<e2>","").replace(
                "</e2>", "")
            template = templates[2].replace("##entity_1##", entity_1).replace("##entity_2##", entity_2)

            sent_feature = tokenizer(
                [(template, sample_sentence)],
                return_token_type_ids=True,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors='pt'
            )
            input_ids = sent_feature['input_ids'].to(device)
            attention_mask = sent_feature['attention_mask'].to(device)
            if 'token_type_ids' in sent_feature.keys():
                token_type_ids = sent_feature['token_type_ids'].to(device)
            else:
                token_type_ids = None
            sent_emb = model.encode(input_ids, attention_mask, token_type_ids=token_type_ids, use_cls=False)  # mark

            sent_emb = sent_emb.detach().cpu().numpy()
            sent_embs.append(sent_emb[0])

    validation_time = format_time(time.time() - t0)
    print("Validation took: {:}".format(validation_time))

    sent_embs = torch.tensor(sent_embs)
    np.save('{}_{}_sent_emb_{}.npy'.format(dataset_name, unseen_nums, i), sent_embs.cpu().numpy())
    print("data dimension is {}. ".format(sent_embs.shape[-1]))
    clusters = KMeans(n_clusters=unseen_nums, n_init=20)  # kmeans
    predict_labels = clusters.fit_predict(sent_embs)

    # evaluation
    metric_result = standard_metric_result(args, test_label_names, predict_labels, label_list)
    # B3
    print('pretrained class eval')
    cluster_eval = ClusterEvaluation(test_labels, predict_labels).printEvaluation()
    print('B3', cluster_eval)
    # NMI, ARI, V_measure
    nmi = normalized_mutual_info_score
    print('NMI', nmi(test_labels, predict_labels))
    print('ARI', adjusted_rand_score(test_labels, predict_labels))
    print('Homogeneity', homogeneity_score(test_labels, predict_labels))
    print('Completeness', completeness_score(test_labels, predict_labels))
    print('V_measure', v_measure_score(test_labels, predict_labels))

    B3_F1 = cluster_eval['F1']
    B3_precision = cluster_eval['precision']
    B3_recall = cluster_eval['recall']
    NMI = normalized_mutual_info_score(test_labels, predict_labels)
    ARI = adjusted_rand_score(test_labels, predict_labels)
    Homogeneity = homogeneity_score(test_labels, predict_labels)
    Completeness = completeness_score(test_labels, predict_labels)
    V_measure = v_measure_score(test_labels, predict_labels)

    evaluation_dict = {
        'test_labellist': label_list,
        'B3_F1': B3_F1,
        'B3_precision': B3_precision,
        'B3_recall': B3_recall,
        'NMI': NMI,
        'ARI': ARI,
        'Homogeneity': Homogeneity,
        'Completeness': Completeness,
        'V_measure': V_measure
    }
    evaluation_dict = json.dumps(evaluation_dict, indent=4)
    with open(save_eval_result, 'w') as f:
        f.write('seen nums: {0}, unseen nums: {1}'.format(seen_nums, unseen_nums))
        f.write('\n')
        f.write(metric_result)
        f.write('\n')
        f.write('\n')
        f.write(evaluation_dict)

def load_data(path, label_path):
    """根据名字加载不同的数据集
    """
    sents = []
    labels = []
    label2id = json.load(open(label_path, 'r'))
    with open(path, 'r') as f:
        for line in f:
            line = json.loads(line)
            label = label2id[line['relation']]
            sents.append((line['origin'], line['origin']))
            labels.append(label)
    return sents, labels, label2id


def load_test(path):
    sents = []
    label_names = []
    label_ids = []
    label_list = []
    if os.path.exists(path):
        json_data = json.load(open(path, 'r'))
        index = 0
        for key in list(json_data.keys()):
            label_list.append(key)
            for item in json_data[key]:
                sent = item['tokens']
                sents.append(sent)
                label_names.append(key)
                label_ids.append(index)
            index += 1
    return sents, label_names, label_ids, label_list



def load_ood_test(path, label_list):
    sentences = []
    labels = []
    label_ids = []
    label_list.append('OOD')
    if os.path.exists(path):
        json_data = json.load(open(path, 'r'))
        index = 0
        for item in json_data:
            sent = item['sentence']
            sentences.append(sent)
            labels.append('OOD')
            label_ids.append(len(label_list))
            index += 1
    return sentences, labels, label_ids, label_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--use_cls", action='store_true', default=False)
    parser.add_argument("--max_length", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--encoder_output_size", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, default=2024)
    parser.add_argument("--dataset", type=str, default='FewRel')
    # parser.add_argument("--data_path", type=str, default='../data/fewrel')

    # setting
    parser.add_argument("--bert_model", type=str, default='bert-base-uncased')
    parser.add_argument("--unseen_nums", type=int, default=30)
    parser.add_argument("--shot", type=str, default='full')
    parser.add_argument("--save_result", type=str, default='output/')

    # main
    args = parser.parse_args()
    args.bert_model = '../pretrained_bert'
    if args.bert_model == 'roberta-base' or args.bert_model == 'roberta-large':
        args.max_length = 144
    if args.dataset == 'tacred':
        args.max_length = 150
    dataset_name = args.dataset
    max_length = args.max_length
    batch_size = args.batch_size
    epochs = args.epochs
    temp = args.temp  # temperature hyper-parameter
    alpha = args.alpha
    dropout = args.dropout
    seeds = args.seeds
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    # data_path = args.data_path
    model_name = args.bert_model
    gpu = args.gpu
    pretrained_model = args.bert_model

    if args.dataset == 'FewRel':
        DATA_TRAIN = f'../datasets/{args.dataset}/{args.unseen_nums}-way-{args.shot}-shot/train.txt'
        DATA_VAL = f'../datasets/{args.dataset}/{args.unseen_nums}-way-{args.shot}-shot/val.txt'
        LABELID_TRAIN = f'../datasets/{args.dataset}/{args.unseen_nums}-way-{args.shot}-shot/trainrel2id.json'
    elif args.dataset == 'wikizsl':
        DATA_TRAIN = f'../datasets/{args.dataset}/{args.unseen_nums}-way-{args.shot}-shot/train.txt'
        DATA_VAL = f'../datasets/{args.dataset}/{args.unseen_nums}-way-{args.shot}-shot/val.txt'
        LABELID_TRAIN = f'../datasets/{args.dataset}/{args.unseen_nums}-way-{args.shot}-shot/trainrel2id.json'
    elif args.dataset == 'TACRED':
        DATA_TRAIN = f'../datasets/{args.dataset}/{args.unseen_nums}-way-{args.shot}-shot/train.txt'
        DATA_VAL = f'../datasets/{args.dataset}/{args.unseen_nums}-way-{args.shot}-shot/val.txt'
        LABELID_TRAIN = f'../datasets/{args.dataset}/{args.unseen_nums}-way-{args.shot}-shot/trainrel2id.json'
    else:
        raise NotImplementedError

    train_data, train_labels, label2id = load_data(DATA_TRAIN, LABELID_TRAIN)
    test_sents, test_label_names, test_labels, label_list = load_test(DATA_VAL)
    
    unseen_nums = args.unseen_nums
    seen_nums = len(label2id)

    save_model = os.path.join(args.save_result, args.dataset + '_{}way_{}shot.pt'.format(args.unseen_nums, args.shot))
    save_eval_result = os.path.join(args.save_result,
                                    args.dataset + '_eval_{}way_{}shot_result.txt'.format(args.unseen_nums, args.shot))

    # tokenizer initialization & add special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model, never_split=['<e1>', '</e1>', '<e2>', '</e2>'])
    special_tokens_dict = {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>']}  # add special token
    tokenizer.add_special_tokens(special_tokens_dict)

    contrastive_sentence_pairs, contrastive_labels = get_contrastive_data(train_data, train_labels)

    # feature
    contrastive_features = get_contrastive_feature(contrastive_sentence_pairs, tokenizer, max_length)
    contrastive_dataset = RCLdataset(contrastive_features, contrastive_labels)

    encoder = Bert_Encoder(rels_num=len(label2id),
                           device=device,
                           chk_path=args.bert_model,
                           id2name=label2id,
                           tokenizer=tokenizer,
                           init_by_cls=False,
                           config=args)
    moment = None

    model = PromptCL(encoder, args.bert_model, temp, device, len(label2id), dropout=dropout,
                     special_tokenizer=tokenizer)

    # train
    if args.do_train:
        train(contrastive_dataset, model, moment, device, train_batch_size=batch_size, train_epochs=epochs, seeds=seeds,
              save_model=save_model, collate_fn=collate_fn)
    else:
        load_model = os.path.join(args.save_result,
                                  args.dataset + '_{}way_{}shot.pt'.format(40, args.shot))
        test(epochs-1, load_model)
