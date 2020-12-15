import torch
from torch.autograd import Function


class DiceCoeff(Function):
    """Autograd function to compute single sample dice coefficient"""

    def forward(self, input, target):
        """Forward pass - computes dice coefficient as a metric
        Args:
            input (torch.Tensor): input tensor
            target (torch.Tensor): target/ground truth tensor
            
        Returns:
            t (torch.FloatTensor): dice coeffient
        """
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    
    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        """Backward pass - Not really sure why this is needed
        """
        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


    
def compute_dice_metric(input, target):
    """Compute dice coefficient as a metric
    
    Args:
        input (torch.Tensor): input tensor
        target (torch.Tensor): target/ground truth tensor

    Returns:
        score (torch.FloatTensor): dice coeffient
    """
    
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

