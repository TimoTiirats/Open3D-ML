import torch
import torch.nn as nn

from ....datasets.utils import DataProcessing


def filter_valid_label(scores, labels, num_classes, ignored_label_inds, device):
    """Loss functions for semantic segmentation."""
    valid_scores = scores.reshape(-1, num_classes).to(device)
    valid_labels = labels.reshape(-1).to(device)

    ignored_bool = torch.zeros_like(valid_labels, dtype=torch.bool)
    for ign_label in ignored_label_inds:
        ignored_bool = torch.logical_or(ignored_bool,
                                        torch.eq(valid_labels, ign_label))

    valid_idx = torch.where(torch.logical_not(ignored_bool))[0].to(device)

    valid_scores = torch.gather(valid_scores, 0,
                                valid_idx.unsqueeze(-1).expand(-1, num_classes))
    valid_labels = torch.gather(valid_labels, 0, valid_idx)

    # Reduce label values in the range of logit shape
    reducing_list = torch.arange(0, num_classes, dtype=torch.int64)
    inserted_value = torch.zeros([1], dtype=torch.int64)

    for ign_label in ignored_label_inds:
        if ign_label >= 0:

            reducing_list = torch.cat([
                reducing_list[:ign_label], inserted_value,
                reducing_list[ign_label:]
            ], 0)
    valid_labels = torch.gather(reducing_list.to(device), 0,
                                valid_labels.long())

    return valid_scores, valid_labels


class SemSegLoss(object):
    """Loss functions for semantic segmentation."""

    def __init__(self, pipeline, model, dataset, device):
        super(SemSegLoss, self).__init__()
        # weighted_CrossEntropyLoss
        if 'class_weights' in dataset.cfg.keys() and len(
                dataset.cfg.class_weights) != 0:
            class_wt = DataProcessing.get_class_weights(
                dataset.cfg.class_weights)
            weights = torch.tensor(class_wt, dtype=torch.float, device=device)

            self.weighted_CrossEntropyLoss = nn.CrossEntropyLoss(weight=weights)
        else:
            self.weighted_CrossEntropyLoss = nn.CrossEntropyLoss()


class SemSegLossV2(object):
    """Loss functions for multi head semantic segmentation."""

    def __init__(self,
                 num_heads,
                 num_classes,
                 ignored_labels=[],
                 device='cpu',
                 weights=None):
        super(SemSegLossV2, self).__init__()
        # weighted_CrossEntropyLoss
        self.weighted_CrossEntropyLoss = []

        for i in range(num_heads):
            if weights is not None and len(weights[i]) != 0:
                wts = DataProcessing.get_class_weights(weights[i])[0]
                assert len(wts) == num_classes[
                    i], f"num_classes : {num_classes[i]} is not equal to number of class weights : {len(wts)}"
                wts = torch.tensor(wts)
            else:
                wts = torch.ones(num_classes[i])
            wts[ignored_labels[i]] = 0
            wts = wts.to(torch.float).to(device)
            self.weighted_CrossEntropyLoss.append(
                nn.CrossEntropyLoss(weight=wts))
