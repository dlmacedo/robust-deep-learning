import numpy as np
from .scores import get_scores
import sklearn.metrics
from .general import get_outputs_labels_and_metrics
 
 
def find_index_of_nearest(value, array):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_ood_metrics_from_scores(in_data_scores, out_data_scores, fpr=0.05):
    assert in_data_scores['type'] == out_data_scores["type"]

    print("====>>>> getting ood detection metrics from scores <<<<====")
    y_true = np.concatenate((np.ones((in_data_scores['values'].size(0),), dtype=int), np.zeros((out_data_scores['values'].size(0),), dtype=int)), axis=None)
    y_pred = np.concatenate((in_data_scores['values'].cpu(), out_data_scores['values'].cpu()), axis=None)

    fpr_list, tpr_list, fpr_tpr_thresholds = sklearn.metrics.roc_curve(y_true, y_pred)
    tpr_at_given_fpr = tpr_list[find_index_of_nearest(fpr, fpr_list)]
    print("tpr_at_given_fpr:\t", '{:.2f}'.format(100*tpr_at_given_fpr))

    auroc = sklearn.metrics.auc(fpr_list, tpr_list)
    #auroc = sklearn.metrics.roc_auc_score(y_true, y_pred)
    print("auroc:\t\t\t", '{:.2f}'.format(100*auroc))

    precision_in, recall_in, precision_in_recall_in_thresholds = sklearn.metrics.precision_recall_curve(y_true, y_pred)
    auprin = sklearn.metrics.auc(recall_in, precision_in)
    #auprin = sklearn.metrics.average_precision_score(y_true, y_pred)
    print("auprin:\t\t\t", '{:.2f}'.format(100*auprin))

    y_pred = -y_pred
    precision_out, recall_out, precision_out_recall_out_thresholds = sklearn.metrics.precision_recall_curve(y_true, y_pred, pos_label=0)
    auprout = sklearn.metrics.auc(recall_out, precision_out)
    #auprout = sklearn.metrics.average_precision_score(y_true, y_pred, pos_label=0)
    print("auprout:\t\t", '{:.2f}'.format(100*auprout))

    results = {}
    results['score_type'] = in_data_scores['type']
    results['fpr'] = fpr_list
    results['tpr'] = tpr_list
    results['fpr_tpr_thresholds'] = fpr_tpr_thresholds
    results['precision_in'] = precision_in
    results['recall_in'] = recall_in
    results['precision_in_recall_in_thresholds'] = precision_in_recall_in_thresholds
    results['precision_out'] = precision_out
    results['recall_out'] = recall_out
    results['precision_out_recall_out_thresholds'] = precision_out_recall_out_thresholds
    results['tpr_at_given_fpr'] = tpr_at_given_fpr
    results['auroc'] = auroc
    results['auprin'] = auprin
    results['auprout'] = auprout
    return results


def get_ood_metrics(model, in_data_valid_loader, out_data_loader, score_type, fpr=0.05, gpu=None):
    print("====>>>> getting ood detection metrics <<<<====")
    results = get_outputs_labels_and_metrics(model, in_data_valid_loader, gpu=gpu)
    in_data_scores = get_scores(results["outputs"], score_type=score_type)
    results = get_outputs_labels_and_metrics(model, out_data_loader, gpu=gpu)
    out_data_scores = get_scores(results["outputs"], score_type=score_type)
    return get_ood_metrics_from_scores(in_data_scores, out_data_scores, fpr=fpr)


def get_ood_detections(model, inputs, thresholds, fpr="0.05", gpu=None):
    model.cuda(gpu)
    model.eval()
    inputs.cuda(gpu, non_blocking=True)
    detections = model(inputs) > thresholds['values'][fpr]
    results = {}
    results['score_type'] = thresholds['score_type']
    results['values'] = detections
    return results
