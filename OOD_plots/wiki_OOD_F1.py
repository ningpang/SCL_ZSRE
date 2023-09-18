import numpy as np
from numpy import linalg
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
dataset_name = 'wikizsl'
# relations = 15
epochs = 2
# value = 0.8 # FewRel
init_values = 0.5
import warnings
warnings.filterwarnings("ignore")
def loop_ood(value, relations):
    feat = np.load('{}_{}_sent_emb_{}.npy'.format(dataset_name, relations, epochs))
    feat = feat / linalg.norm(feat, ord=2, axis=1, keepdims=True)
    pred = np.load('{}_{}_prediction.npy'.format(dataset_name, relations))
    label = np.load('{}_{}_label.npy'.format(dataset_name, relations))

    f1 = metrics.f1_score(label, pred, average='macro')
    # print(f1)

    new_label = []
    new_pred = []
    label2id = {}
    for i in range(len(label)):
        if label[i] not in label2id:
            label2id[label[i]] = len(label2id)
    for l in label:
        new_label.append(label2id[l])
    for p in pred:
        new_pred.append(label2id[p])
    pred = np.array(new_pred)
    label = np.array(new_label)
    # print(metrics.f1_score(label, pred, average='macro'))

    centers = {}
    for i, l in enumerate(pred):
        if l not in centers:
            centers[l] = []
        centers[l].append(feat[i])

    proto = {}
    for i in centers:
        cen = np.stack(centers[i]).mean(axis=0)
        proto[i] = cen

    # print(proto.keys())

    def L2_dis(x, y):
        return linalg.norm(x - y, ord=2, axis=0)

    correct = []
    wrong = []
    score = []
    all = []
    for i in range(feat.shape[0]):
        dis = L2_dis(feat[i], proto[pred[i]])
        score.append(dis)
        all.append(dis)
        if pred[i] == label[i]:
            correct.append(dis)
        else:
            wrong.append(dis)
    all.sort()
    # print(init_values)
    bound = all[int(init_values * len(all))-1]
    # plt.figure()
    # plt.axvline(x=bound, color='r')
    # sns.kdeplot(correct)
    # sns.kdeplot(wrong)
    # plt.show()

    new_preds = []
    new_labels = []

    for i in range(len(pred)):
        p = pred[i]
        l = label[i]
        s = score[i]
        if s <= bound:
            new_labels.append(l)
            new_preds.append(p)
        # else:
        #     new_labels.append(l)
        #     new_preds.append(relations)
    metric_result = classification_report(new_labels, new_preds, digits=5)
    final_f1 = metrics.f1_score(new_labels, new_preds, average='macro')

    # print(metric_result)
    from evaluation import ClusterEvaluation
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, \
        completeness_score, \
        v_measure_score
    import os, json
    # print('pretrained class eval')
    test_labels = new_labels
    predict_labels = new_preds
    cluster_eval = ClusterEvaluation(test_labels, predict_labels).printEvaluation()
    # print('B3', cluster_eval)
    # NMI, ARI, V_measure
    nmi = normalized_mutual_info_score
    # print('NMI', nmi(test_labels, predict_labels))
    # print('ARI', adjusted_rand_score(test_labels, predict_labels))
    # print('Homogeneity', homogeneity_score(test_labels, predict_labels))
    # print('Completeness', completeness_score(test_labels, predict_labels))
    # print('V_measure', v_measure_score(test_labels, predict_labels))

    B3_F1 = cluster_eval['F1']
    B3_precision = cluster_eval['precision']
    B3_recall = cluster_eval['recall']
    NMI = normalized_mutual_info_score(test_labels, predict_labels)
    ARI = adjusted_rand_score(test_labels, predict_labels)
    Homogeneity = homogeneity_score(test_labels, predict_labels)
    Completeness = completeness_score(test_labels, predict_labels)
    V_measure = v_measure_score(test_labels, predict_labels)

    evaluation_dict = {
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
    save_eval_result = os.path.join('../output/',
                                    '{}_eval_{}way_{}shot_result_{}_ood.txt'.format(dataset_name, relations, 'full',
                                                                                    round(value,2)))

    with open(save_eval_result, 'w') as f:
        f.write('\n')
        f.write(metric_result)
        f.write('\n')
        f.write('\n')
        f.write(evaluation_dict)

    return final_f1
import seaborn as sns
sns.set(style="ticks", font_scale=1.2)
fig = plt.figure()
# for relations in [20, 30, 35, 40]:
#     f1_scores = []
#     init_values = 0.5
#     while True:
#         if init_values > 1.02:
#             break
#         final_f1 = loop_ood(init_values, relations)
#         f1_scores.append(final_f1)
#         init_values += 0.05
#     f1_scores = [item*100 for item in f1_scores]
#     print(f1_scores)
#     # print(len(np.arange(0.5, 1, 0.05)), len(f1_scores))
#     sns.lineplot(x=np.arange(0.5, 1.05, 0.05), y=f1_scores, marker='^', markersize=12, linestyle='--', linewidth=2.5,
#                  label='Num of Rel: {}'.format(relations))
# plt.ylim(60, 80)
# plt.legend(ncol=2, loc='lower left')
# plt.grid(ls=':', lw=2)
# plt.ylabel('F1 (%)')
# plt.xlabel('Ratio of retention samples')
# plt.title('Wiki-ZSL')
# plt.savefig('wiki_F1.pdf', bbox_inches ="tight")
# plt.show()

ood_20 = [69.29931708313806, 69.83652166852968, 69.84041891059114, 69.78140416863408, 70.00735011340757, 70.018158591262, 69.59707989932365, 69.44498906290664, 69.37746945807956, 69.25870521165803, 68.85203018605026]
ood_30 = [70.94636245595602, 70.63241475854493, 70.08137019575658, 69.72183908910198, 69.1995184591142, 68.76229374722517, 68.33524684145281, 67.89674821544077, 67.50564583968354, 67.15412210063906, 66.73806963647138]
ood_35 = [72.0303638124825, 71.67933018237618, 71.27573312885262, 70.53452110987803, 69.64267354012883, 68.90184003386265, 68.2961534009254, 67.65497134771508, 66.9463837590793, 66.49366661587527, 66.02773359349482]
ood_40 = [75.11918766360304, 73.05715419831556, 72.81527522890654, 72.24248631857303, 71.54438852662794, 70.91132306933117, 69.99303388406125, 68.95207410327033, 68.13437482576516, 67.21120871377313, 66.02338085444333]
ood = [ood_20, ood_30, ood_35, ood_40]
nums = [20, 30, 35, 40]
for idx, rel in enumerate(nums):
    sns.lineplot(x=np.arange(0.5, 1.05, 0.05), y=ood[idx], marker='^', markersize=12, linestyle='--', linewidth=2.5,
                 label='Num of Rel: {}'.format(rel))
plt.ylim(62, 76)
plt.legend(ncol=2, loc='lower left')
plt.grid(ls=':', lw=2)
plt.ylabel('F1 (%)')
plt.xlabel('Ratio of retention samples')
plt.title('Wiki-ZSL')
plt.savefig('wiki_F1.pdf', bbox_inches ="tight")
plt.show()