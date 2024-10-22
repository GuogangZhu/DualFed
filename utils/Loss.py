import torch
import torch.nn as nn
import torch.nn.functional as F

class Contrastive_Loss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1):
        super(Contrastive_Loss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None):
        """Compute loss for model. If both `labels` and `mask` are None, it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, ...].
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        eps = 1e-6

        device = (torch.device('cuda:0') if features.is_cuda else torch.device('cpu'))

        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)

        features_norm = F.normalize(features, dim=1)

        # compute logits
        feature_dot = torch.div(torch.matmul(features_norm, features_norm.T),
                                self.temperature)

        # mask-out self-contrast cases
        # [[0, 1, 1...]
        #  [1, 0, 1...]
        #  [1, 1, 0...]]
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        # element-wise multiple
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(feature_dot) * logits_mask
        log_prob = feature_dot - torch.log(exp_logits.sum(1, keepdims=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + eps)
        loss = -mean_log_prob_pos

        loss = loss.view(batch_size).mean()

        return loss