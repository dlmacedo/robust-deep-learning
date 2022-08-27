import torch.nn as nn
import torch


class IsoMaxLossFirstPart(nn.Module):
    """This part replaces the model classifier output layer nn.Linear()."""
    def __init__(self, num_features, num_classes, temperature=1.0):
        super(IsoMaxLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = nn.Parameter(torch.tensor([temperature]), requires_grad=False)        
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        nn.init.constant_(self.prototypes, 0.0)

    def forward(self, features):
        distances = torch.cdist(features, self.prototypes, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        logits = -distances
        return logits / self.temperature

    def extra_repr(self):
        return 'num_features={}, num_classes={}'.format(self.num_features, self.num_classes)


class IsoMaxLossSecondPart(nn.Module):
    """This part replaces the nn.CrossEntropyLoss()"""
    def __init__(self, model_classifier, debug=False, gpu=None):
        super(IsoMaxLossSecondPart, self).__init__()
        self.model_classifier = model_classifier
        self.entropic_scale = 10.0
        self.debug = debug
        self.gpu = gpu

    def preprocess(self, inputs, targets):
        return inputs, targets

    def forward(self, logits, targets):
        ##############################################################################
        ##############################################################################
        """Probabilities and logarithms are calculated separately and sequentially."""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss."""
        ##############################################################################
        ##############################################################################
        num_classes = logits.size(1)
        targets_one_hot = torch.eye(num_classes)[targets].long().cuda(self.gpu)        
        probabilities_for_training = nn.Softmax(dim=1)(self.entropic_scale * logits)
        probabilities_at_targets = probabilities_for_training[range(logits.size(0)), targets]
        loss = -torch.log(probabilities_at_targets).mean()

        if not self.debug:
            return loss
        else:
            intra_inter_logits = torch.where(targets_one_hot != 0, logits, torch.Tensor([float('Inf')]).cuda(self.gpu))
            inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(self.gpu), logits)
            intra_logits = intra_inter_logits[intra_inter_logits != float('Inf')].detach().cpu().numpy()
            inter_logits = inter_intra_logits[inter_intra_logits != float('Inf')].detach().cpu().numpy()
            return loss, 1.0, inter_logits, intra_logits
