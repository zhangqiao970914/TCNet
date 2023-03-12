import torch
import torch.nn.functional as F
"""
IoU_loss:
    Compute IoU loss between predictions and ground-truths for training [Equation 3].
"""
"""
IoU_loss:
    Compute IoU loss between predictions and ground-truths for training [Equation 3].
"""
def IoU_loss(preds_list, gt):
    preds = torch.cat(preds_list, dim=1)
    N, C, H, W = preds.shape
    min_tensor = torch.where(preds < gt, preds, gt)    # shape=[N, C, H, W]
    max_tensor = torch.where(preds > gt, preds, gt)    # shape=[N, C, H, W]
    min_sum = min_tensor.view(N, C, H * W).sum(dim=2)  # shape=[N, C]
    max_sum = max_tensor.view(N, C, H * W).sum(dim=2)  # shape=[N, C]
    loss = 1 - (min_sum / max_sum).mean()
    return loss


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()



