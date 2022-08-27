import torch.nn as nn
import torch
import math


class SoftMaxLossFirstPart(nn.Module):
    def __init__(self, num_features, num_classes, temperature=1.0):
        super(SoftMaxLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = nn.Parameter(torch.tensor([temperature]), requires_grad=False)
        self.weights = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.bias = nn.Parameter(torch.Tensor(num_classes))
        nn.init.uniform_(self.weights, a=-math.sqrt(1.0/self.num_features), b=math.sqrt(1.0/self.num_features))
        nn.init.zeros_(self.bias)

    def forward(self, features):
        logits = features.matmul(self.weights.t()) + self.bias        
        return logits / self.temperature

    def extra_repr(self):
        return 'num_features={}, num_classes={}, bias={}'.format(self.num_features, self.num_classes, self.bias is not None)


class SoftMaxLossSecondPart(nn.Module):
    def __init__(self, model_classifier, debug=False, gpu=None):
        super(SoftMaxLossSecondPart, self).__init__()
        self.model_classifier = model_classifier
        self.loss = nn.CrossEntropyLoss()
        self.debug = debug
        self.gpu = gpu

    def preprocess(self, inputs, targets):
        return inputs, targets

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        targets_one_hot = torch.eye(num_classes)[targets].long().cuda(self.gpu)
        loss = self.loss(logits, targets)

        if not self.debug:
            return loss
        else:
            intra_inter_logits = torch.where(targets_one_hot != 0, logits, torch.Tensor([float('Inf')]).cuda(self.gpu))
            inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(self.gpu), logits)
            intra_logits = intra_inter_logits[intra_inter_logits != float('Inf')].detach().cpu().numpy()
            inter_logits = inter_intra_logits[inter_intra_logits != float('Inf')].detach().cpu().numpy()
            return loss, 1.0, inter_logits, intra_logits
