from __future__ import absolute_import
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        """
        Args:
            input: feature matrix with shape (batch_size, num_classes).maybe is the logits(will tackle with softmax function)
            labels: ground truth labels with shape (batch_size).
        """
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class GHMC(nn.Module):
    def __init__(
            self,
            bins=10,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def _expand_binary_labels(self, labels, label_weights, label_channels):
        # expand labels
        bin_labels = labels.new_full((labels.size(0), label_channels), 0)
        # inds = torch.nonzero(labels >= 1).squeeze()
        inds = torch.nonzero(labels >= 0).squeeze()
        if inds.numel() > 0:
            # bin_labels[inds, labels[inds] - 1] = 1
            bin_labels[inds, labels[inds]] = 1
        # expand label_weights(label_weight should with size [batch_num], otherwise the function "expand" cannot work)
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
        return bin_labels, bin_label_weights

    def forward(self, pred, target, label_weight_input=None, *args, **kwargs):
        """Calculate the GHM-C loss.
        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num] or size [batch_num]):
                Binary class target for each sample.
                if size [batch_num], it will be expand with function "_expand_binary_labels"
            label_weight (with same size to target.float tensor of size [batch_num, class_num] or size [batch_num]):
                the value is 1 if the sample is valid and 0 if ignored.
                if size [batch_num], it will be expand with function "_expand_binary_labels"
        Returns:
            The gradient harmonized loss.
        """
        if label_weight_input == None:
            label_weight = torch.ones_like(target)
        else:
            label_weight = label_weight_input

        # the target should be binary class label
        if pred.dim() != target.dim():
            target, label_weight = self._expand_binary_labels(
                                    target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        # weights = torch.pow(weights, 0.5)

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight
