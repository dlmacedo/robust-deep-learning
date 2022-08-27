import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math


class DisMaxLossFirstPart(nn.Module):
    """This part replaces the model classifier output layer nn.Linear()."""
    def __init__(self, num_features, num_classes, temperature=1.0):
        super(DisMaxLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.distance_scale = nn.Parameter(torch.Tensor(1)) 
        nn.init.constant_(self.distance_scale, 1.0)
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        self.temperature = nn.Parameter(torch.tensor([temperature]), requires_grad=False)        

    def forward(self, features):
        distances_from_normalized_vectors = torch.cdist(
            F.normalize(features), F.normalize(self.prototypes), p=2.0, compute_mode="donot_use_mm_for_euclid_dist") / math.sqrt(2.0)    
        isometric_distances = torch.abs(self.distance_scale) * distances_from_normalized_vectors
        logits = -(isometric_distances + isometric_distances.mean(dim=1, keepdim=True))
        return logits / self.temperature

    def extra_repr(self):
        return 'num_features={}, num_classes={}'.format(self.num_features, self.num_classes)


class DisMaxLossSecondPart(nn.Module):
    """This part replaces the nn.CrossEntropyLoss()"""
    def __init__(self, model_classifier, add_on=None, debug=False, gpu=None):
        super(DisMaxLossSecondPart, self).__init__()
        self.model_classifier = model_classifier
        self.entropic_scale = 10.0
        self.add_on = add_on
        self.alpha = 1.0
        self.debug = debug
        self.gpu = gpu

    def preprocess(self, inputs, targets):
        batch_size = inputs.size(0)
        half_batch_size = batch_size//2
        W = inputs.size(2)
        H = inputs.size(3)
        idx = torch.randperm(batch_size)
        inputs = inputs[idx].view(inputs.size())
        targets = targets[idx].view(targets.size())
        if self.add_on == "fpr":
            print("fpr1")
            inputs[half_batch_size:, :, W//2:, :H//2] = torch.roll(inputs[half_batch_size:, :, W//2:, :H//2], 1, 0)
            inputs[half_batch_size:, :, :W//2, H//2:] = torch.roll(inputs[half_batch_size:, :, :W//2, H//2:], 2, 0)
            inputs[half_batch_size:, :, W//2:, H//2:] = torch.roll(inputs[half_batch_size:, :, W//2:, H//2:], 3, 0)
        return inputs, targets

    def forward(self, logits, targets):
        ##############################################################################
        ##############################################################################
        """Probabilities and logarithms are calculated separately and sequentially."""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss."""
        ##############################################################################
        ##############################################################################
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        half_batch_size = batch_size//2
        targets_one_hot = torch.eye(num_classes)[targets].long().cuda(self.gpu)

        if self.model_classifier.training:
            probabilities_for_training = nn.Softmax(dim=1)(self.entropic_scale * logits)
            if self.add_on is None: # no add_on
                probabilities_at_targets = probabilities_for_training[range(batch_size), targets]
                loss = -torch.log(probabilities_at_targets).mean()
            else: # add_on                                
                probabilities_at_targets = probabilities_for_training[range(half_batch_size), targets[:half_batch_size]]
                loss = -torch.log(probabilities_at_targets).mean()
                if self.add_on == "fpr":
                    print("fpr2")
                    targets_one_hot_0 = torch.eye(num_classes)[torch.roll(targets[half_batch_size:], 0, 0)].long().cuda(self.gpu)
                    targets_one_hot_1 = torch.eye(num_classes)[torch.roll(targets[half_batch_size:], 1, 0)].long().cuda(self.gpu)
                    targets_one_hot_2 = torch.eye(num_classes)[torch.roll(targets[half_batch_size:], 2, 0)].long().cuda(self.gpu)
                    targets_one_hot_3 = torch.eye(num_classes)[torch.roll(targets[half_batch_size:], 3, 0)].long().cuda(self.gpu)
                    target_distributions_0 = torch.where(targets_one_hot_0 != 0, torch.tensor(0.25).cuda(self.gpu), torch.tensor(0.0).cuda(self.gpu))
                    target_distributions_1 = torch.where(targets_one_hot_1 != 0, torch.tensor(0.25).cuda(self.gpu), torch.tensor(0.0).cuda(self.gpu))
                    target_distributions_2 = torch.where(targets_one_hot_2 != 0, torch.tensor(0.25).cuda(self.gpu), torch.tensor(0.0).cuda(self.gpu))
                    target_distributions_3 = torch.where(targets_one_hot_3 != 0, torch.tensor(0.25).cuda(self.gpu), torch.tensor(0.0).cuda(self.gpu))
                    target_distributions_total = target_distributions_0 + target_distributions_1 + target_distributions_2 + target_distributions_3
                    loss_add_on = F.kl_div(torch.log(probabilities_for_training[half_batch_size:]), target_distributions_total, reduction='batchmean')
                else:
                    sys.exit('You should pass a valid add on!!!')
                loss = (loss + (self.alpha * loss_add_on))/2

        else: # validation
            probabilities_for_inference = nn.Softmax(dim=1)(logits)
            probabilities_at_targets = probabilities_for_inference[range(batch_size), targets]
            loss = -torch.log(probabilities_at_targets).mean()

        if not self.debug:
            return loss
        else:
            intra_inter_logits = torch.where(targets_one_hot != 0, logits, torch.Tensor([float('Inf')]).cuda(self.gpu))
            inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(self.gpu), logits)
            intra_logits = intra_inter_logits[intra_inter_logits != float('Inf')].detach().cpu().numpy()
            inter_logits = inter_intra_logits[inter_intra_logits != float('Inf')].detach().cpu().numpy()
            return loss, self.model_classifier.distance_scale.item(), inter_logits, intra_logits

