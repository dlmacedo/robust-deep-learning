import sys
import torch


def get_scores(logits, score_type):
    print("====>>>> getting scores <<<<====")
    if score_type == "MPS": # the maximum probability score
        probabilities = torch.nn.Softmax(dim=1)(logits)
        scores = probabilities.max(dim=1)[0]
    elif score_type == "ES": # the negative entropy score
        probabilities = torch.nn.Softmax(dim=1)(logits)
        scores = (probabilities * torch.log(probabilities)).sum(dim=1)
    elif score_type == "MDS": # the minimum distance score
        scores = logits.max(dim=1)[0]
    elif score_type == "MMLS": # the max-mean logit score
        scores = logits.max(dim=1)[0] + logits.mean(dim=1)
    elif score_type == "MMLES": # the max-mean logit entropy score
        probabilities = torch.nn.Softmax(dim=1)(logits)
        scores = logits.max(dim=1)[0] + logits.mean(dim=1) + (probabilities * torch.log(probabilities)).sum(dim=1)
    elif score_type == "MMLEPS": # the max-mean logit entropy probability score
        probabilities = torch.nn.Softmax(dim=1)(logits)
        scores = logits.max(dim=1)[0] + logits.mean(dim=1) + (probabilities * torch.log(probabilities)).sum(dim=1) + probabilities.max(dim=1)[0]
    else:
        sys.exit('You should use a valid score type!!!')

    results = {}
    results['type'] = score_type
    results['values'] = scores
    return results
