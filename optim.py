import torch.optim as optim

def get_optimizer(network, lr, lr_milestone, lr_gamma):
    optimizer = optim.Adam(network.parameters(), lr=lr)

    scheduler = None
    if lr_milestone is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer   = optimizer, 
                                                   milestones  = lr_milestone,
                                                   gamma       = lr_gamma)

    return optimizer, scheduler
