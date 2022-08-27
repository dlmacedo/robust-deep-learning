import torch
import numpy
from tqdm import tqdm
from .general import get_outputs_labels_and_metrics
from .scores import get_scores


def get_thresholds_from_scores(scores):
    print("====>>>> getting thresholds <<<<====")
    thresholds = torch.Tensor(25)
    thresholds = {}
    for percentile in tqdm([0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]):
        thresholds[str(percentile/100)] = numpy.percentile(scores, percentile)
    results = {}
    results['score_type'] = scores['type']
    results['values'] = thresholds
    return results


def get_thresholds(model, in_data_val_loader, score_type, gpu=None):
    results = get_outputs_labels_and_metrics(model, in_data_val_loader, gpu=gpu)
    in_data_scores = get_scores(results["outputs"], score_type)
    return get_thresholds_from_scores(in_data_scores)
