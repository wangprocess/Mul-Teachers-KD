import numpy
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score#, sensitivity_score
from sklearn.metrics import roc_auc_score

N_CLASSES = 3
CLASS_NAMES = ['BENIGN', 'MALIGNANT', 'NORMAL']


def mean_accuracy(scores, labels):
    prob = torch.softmax(scores, dim=1)
    equal = (labels.data == prob.max(dim=1)[1])
    size_batch = prob.size(0)
    acc = (equal.sum()).float()/size_batch
    print(acc)
    return acc

"""
    The evaluation metrics are calculated With reference to Semi-Supervised Medical Image Classification with Relation-Driven Self-Ensembling Mode
    We make a fair comparison
    The link of github:  https://github.com/liuquande/SRC-MT
"""

def compute_metrics_test(gt, pred, competition=True):
    """
    Computes accuracy, precision, recall and F1-score from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
        competition: whether to use competition tasks. If False,
          use all tasks
    Returns:
        List of AUROCs of all classes.
    """

    AUROCs, Accus, Senss, Specs, Pre, F1 = [], [], [], [], [], []
    gt_np = numpy.array(gt)
    # if cfg.uncertainty == 'U-Zeros':
    #     gt_np[np.where(gt_np==-1)] = 0
    # if cfg.uncertainty == 'U-Ones':
    #     gt_np[np.where(gt_np==-1)] = 1
    pred_np = numpy.array(pred)
    THRESH = 0.33
    #     indexes = TARGET_INDEXES if competition else range(N_CLASSES)
    # indexes = range(n_classes)

    #     pdb.set_trace()
    indexes = range(len(CLASS_NAMES))

    for i, cls in enumerate(indexes):
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        except ValueError as error:
            print('Error in computing accuracy for {}.\n Error msg:{}'.format(i, error))
            AUROCs.append(0)

        try:
            Accus.append(accuracy_score(gt_np[:, i], (pred_np[:, i] >= THRESH)))
        except ValueError as error:
            print('Error in computing accuracy for {}.\n Error msg:{}'.format(i, error))
            Accus.append(0)

        try:
            Pre.append(precision_score(gt_np[:, i], (pred_np[:, i] >= THRESH)))
        except ValueError:
            print('Error in computing F1-score for {}.'.format(i))
            Pre.append(0)

        try:
            F1.append(f1_score(gt_np[:, i], (pred_np[:, i] >= THRESH)))
        except ValueError:
            print('Error in computing F1-score for {}.'.format(i))
            F1.append(0)

    return AUROCs, Accus, Senss, Specs, Pre, F1