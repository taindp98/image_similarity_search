from torch import nn
import torch

class TripletLoss(nn.Module):
    """
    Triplet loss
    """
    def __init__(self, alpha, device='cuda'):
        super(TripletLoss, self).__init__()
        ## alpha --> bias
        self.alpha = alpha
        self.device = device

    def forward(self, anchor, positive, negative, average_loss=True):
        ## Frobenius norm
        d_p = torch.norm(anchor - positive,dim=1)
        d_n = torch.norm(anchor - negative,dim=1)

        losses = torch.max(d_p - d_n + self.alpha, torch.FloatTensor([0]).to(self.device))
        
        if average_loss:
            return losses.mean(), d_p.mean(), d_n.mean()

        return losses.sum(), d_p.mean(), d_n.mean()