import numpy as np
from numpy import linalg
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
dataset_name = 'FewRel'
# relations = 15
epochs = 3
# value = 0.8 # FewRel
init_values = 0.5
import warnings
warnings.filterwarnings("ignore")
def loop_ood(value, relations):
    feat = np.load('../{}_{}_sent_emb_{}.npy'.format(dataset_name, relations, epochs))
    feat = feat / linalg.norm(feat, ord=2, axis=1, keepdims=True)
    pred = np.load('../{}_{}_prediction.npy'.format(dataset_name, relations))
    label = np.load('../{}_{}_label.npy'.format(dataset_name, relations))

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
    from MPRCL.evaluation import ClusterEvaluation
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, \
        completeness_score, \
        v_measure_score
    import os, json
    # print('pretrained class eval')
    test_labels = new_labels
    predict_labels = new_preds
    cluster_eval = ClusterEvaluation(test_labels, predict_labels).printEvaluation()
    # print('B3', cluster_eval)
    # # NMI, ARI, V_measure
    # nmi = normalized_mutual_info_score
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
#     # print(len(np.arange(0.5, 1, 0.05)), len(f1_scores))
#     f1_scores = [item*100 for item in f1_scores]
#     print(f1_scores)
#     sns.lineplot(x=np.arange(0.5, 1.05, 0.05), y=f1_scores, marker='^', markersize=12, linestyle='--', linewidth=2.5,
#                  label='Num of Rel: {}'.format(relations))
# plt.ylim(65, 90)
# plt.legend(ncol=2, loc='lower left')
# plt.grid(ls=':', lw=2)
# plt.ylabel('F1(%)')
# plt.xlabel('Ratio of retention samples')
# plt.title(dataset_name)
# plt.savefig('FewRel_F1.pdf', bbox_inches ="tight")
# plt.show()

ood_20 = [88.20878164338619, 88.00366503589396, 87.86075561285409, 87.78901388913741, 87.76793973976659, 89.49286598927915, 90.23288283712748, 89.22982626140558, 87.71669039318907, 86.35862565072635, 84.61816334848507]
ood_30 = [75.17912858131616, 75.97893291616857, 77.08451787721631, 78.23103026797628, 78.56871934643601, 78.96906979760037, 79.41058890294921, 78.9905080229072, 78.36674639490582, 77.01150233111939, 75.18076210357489]
ood_35 = [74.64343978197138, 74.85010272513944, 74.78516921138467, 74.87023621178336, 75.69193619889741, 76.34945277684561, 76.26016454642503, 75.74761714463533, 74.71864122109942, 73.3930722510377, 71.98946293769795]
ood_40 = [77.5727084177394, 77.54344668146038, 77.76894872788696, 78.81108117272323, 79.60862146000973, 79.21350530529297, 78.86912590452879, 77.89480574186352, 76.58132447471542, 75.10064917517344, 73.35796401112319]
nums = [20, 30, 35, 40]
ood = [ood_20, ood_30, ood_35, ood_40]
for idx, rel in enumerate(nums):
    sns.lineplot(x=np.arange(0.5, 1.05, 0.05), y=ood[idx], marker='^', markersize=12, linestyle='--', linewidth=2.5,
                 label='Num of Rel: {}'.format(rel))
plt.ylim(65, 93)
plt.legend(ncol=2, loc='lower left')
plt.grid(ls=':', lw=2)
plt.ylabel('F1(%)')
plt.xlabel('Ratio of retention samples')
plt.title(dataset_name)
plt.savefig('FewRel_F1.pdf', bbox_inches ="tight")
plt.show()
