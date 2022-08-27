
import torch
from tqdm import tqdm
from .uncertainty import ECELoss


def get_outputs_labels_and_metrics(model, loader, gpu=None):
    print("====>>>> getting metrics, outputs, and labels <<<<====")
    correct = 0
    total = 0
    outputs_list = []
    labels_list = []
    model.cuda(gpu)
    model.eval()
    ece_criterion = ECELoss()
    nll_criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, labels in tqdm(loader):
            data = data.cuda(gpu, non_blocking=True)
            labels = labels.cuda(gpu, non_blocking=True)
            outputs = model(data)
            outputs_list.append(outputs)
            labels_list.append(labels)
            output_probs = torch.nn.functional.softmax(outputs,dim=1)
            max_probs, predicted = torch.max(output_probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        outputs = torch.cat(outputs_list)
        labels = torch.cat(labels_list)
    acc = 100 * correct / total
    nll = nll_criterion(outputs, labels).item()
    ece = ece_criterion.loss(outputs.cpu().numpy(),labels.cpu().numpy(),15)

    results = {}
    results['outputs'] = outputs
    results['labels'] = labels
    results['acc'] = acc
    results['nll'] = nll
    results['ece'] = ece
    return results
