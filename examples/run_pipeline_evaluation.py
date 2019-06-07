##################################################
# Assumes: examples/run_pipeline.py is run first
##################################################
import pickle
from eval.evaluation import *
from sklearn.preprocessing import OneHotEncoder
from itertools import cycle

ohe = OneHotEncoder(sparse=False)

image_set_dir = 'mm_e16.5_20x_sox9_sftpc_acta2/light_color_corrected'
image_set_path = os.path.join('data', image_set_dir)
output_path = os.path.join(
    'tmp',
    '_'.join([image_set_dir, 'pipeline'])
)

with open(os.path.join(output_path, 'xgb_model.pkl'), 'rb') as f:
    eval_data = pickle.load(f)

labels = list(set(x['label'] for x in eval_data['truth']['regions']))
ohe.fit(np.array(labels).reshape(-1,1))


iou_mat, pred_mat = generate_iou_pred_matrices(
    eval_data['truth'],
    eval_data['predictions']
)

# I need to calculate the AUC for each class, in order to do so, I need to line up predictions and one hot encoded
# labels side by side. To do this, I need to order the
y_truth = []
y_pred = []
for pred_ind in range(len(eval_data['predictions'])):
    if np.sum(iou_mat[pred_ind,:])>0:
        truth_ind = np.argmax(iou_mat[pred_ind,:])
        y_pred.append(
            np.array(pd.DataFrame([eval_data['predictions'][pred_ind]['label']['prob']])[ohe.categories_[0].tolist()])[0]
        )
        y_truth.append(ohe.transform(np.array(eval_data['truth']['regions'][truth_ind]['label']).reshape(-1, 1))[0])
    else:
        y_pred.append(
            np.array(pd.DataFrame([eval_data['predictions'][pred_ind]['label']['prob']])[ohe.categories_[0].tolist()])[0]
        )
        y_truth.append(ohe.transform(np.array('background').reshape(-1, 1))[0])

y_truth_array = np.stack(y_truth)
y_pred_array = np.stack(y_pred)


from sklearn.metrics import roc_curve, auc


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(y_truth_array[:, i], y_pred_array[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

lw=2

for i, x in enumerate(ohe.categories_[0]):
    fpr[x] = fpr.pop(i)
    tpr[x] = tpr.pop(i)
    roc_auc[x] = roc_auc.pop(i)

plt.figure()

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'purple', 'green', 'red'])
for i, color in zip(list(fpr.keys()), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='{0} (auc={1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Process Residual (False), Predict Model (None) ROC')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(6):
    precision[i], recall[i], _ = precision_recall_curve(y_truth_array[:, i],
                                                        y_pred_array[:, i])
    average_precision[i] = average_precision_score(y_truth_array[:, i], y_pred_array[:, i])

for i, x in enumerate(ohe.categories_[0]):
    precision[x] = precision.pop(i)
    recall[x] = recall.pop(i)
    average_precision[x] = average_precision.pop(i)


f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
plt.figure(figsize=(7, 8))
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

for i, color in zip(list(fpr.keys()), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('{0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))
fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Process Residual (False), Predict Model (None) Precision Recall')
plt.legend(lines, labels, loc=(0, -.58), prop=dict(size=14))
plt.show()