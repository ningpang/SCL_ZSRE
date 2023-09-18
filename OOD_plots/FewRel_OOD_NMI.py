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

    return NMI
import seaborn as sns
sns.set(style="ticks", font_scale=1.2)
fig = plt.figure()
# for relations in [20, 30, 35, 40]:
#     NMI_scores = []
#     init_values = 0.5
#     while True:
#         if init_values > 1.02:
#             break
#         final_NMI = loop_ood(init_values, relations)
#         NMI_scores.append(final_NMI)
#         init_values += 0.05
#     NMI_scores = [item*100 for item in NMI_scores]
#     print(NMI_scores)
#     # print(len(np.arange(0.5, 1, 0.05)), len(f1_scores))
#     sns.lineplot(x=np.arange(0.5, 1.05, 0.05), y=NMI_scores, marker='^', markersize=12, linestyle='--', linewidth=2.5,
#                  label='Num of Rel: {}'.format(relations))
#
#
# plt.ylim(80, 100)
# plt.legend(ncol=2, loc = 'lower left')
# plt.grid(ls=':', lw=2)
# plt.ylabel('NMI')
# plt.xlabel('Ratio of retention samples')
# plt.title(dataset_name)
# plt.savefig('FewRel_NMI.pdf', bbox_inches ="tight")
# plt.show()

ood_20 = [96.19986611537497, 96.01025543794309, 95.69604232562368, 95.55454558945638, 95.27317731563807, 94.79830588578326, 93.83406537179143, 92.82695572079369, 91.49608987798317, 89.71279018390747, 86.3392740192896]
ood_30 = [94.04140092781198, 93.59056595733865, 93.32255711035445, 92.95252334940875, 92.67536754181397, 92.06518616517194, 91.38654342462738, 90.31422459936337, 89.09406287006291, 87.08943873620719, 84.25082764048321]
ood_35 = [93.35878529969747, 93.0044613672565, 92.74003844422126, 92.49481665244535, 92.05653704149404, 91.48021903608806, 90.50057202711908, 89.06287364291862, 87.26741636463075, 85.20254874797547, 82.56006238332849]
ood_40 = [92.6727315540608, 92.40348243062576, 92.16565017372938, 91.80675673045961, 91.32923589456982, 90.41769328111533, 89.49075887896345, 88.48954231940156, 87.07910677274738, 85.12959229944606, 82.39488018657899]

ood = [ood_20, ood_30, ood_35, ood_40]
nums = [20, 30, 35, 40]
for idx, rel in enumerate(nums):
    sns.lineplot(x=np.arange(0.5, 1.05, 0.05), y=ood[idx], marker='^', markersize=12, linestyle='--', linewidth=2.5,
                 label='Num of Rel: {}'.format(rel))
plt.ylim(80, 100)
plt.legend(ncol=2, loc = 'lower left')
plt.grid(ls=':', lw=2)
plt.ylabel('NMI (%)')
plt.xlabel('Ratio of retention samples')
plt.title(dataset_name)
plt.savefig('FewRel_NMI.pdf', bbox_inches ="tight")
plt.show()
