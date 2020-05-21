import torch
# the guided loss for retinanet to balance cls loss and reg loss
def GuidedLoss(loss_box_cls,loss_box_reg):
    with torch.no_grad():
        w=loss_box_reg/loss_box_cls
    loss_box_cls*=w
    return loss_box_cls