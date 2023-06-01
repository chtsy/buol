from torch import nn

L1Loss = nn.L1Loss
MSELoss = nn.MSELoss
CrossEntropyLoss = nn.CrossEntropyLoss
BCELoss = nn.BCELoss

def loss(cfg):
    if cfg.NAME == 'CrossEntropyLoss':
        return CrossEntropyLoss(ignore_index=cfg.IGNORE, reduction='none')
    elif cfg.NAME == 'MSELoss':
        return MSELoss(reduction='none')
    elif cfg.NAME == 'L1Loss':
        return L1Loss(reduction='none')
    elif cfg.NAME == 'BCELoss':
        return BCELoss(reduction='none')
    else:
        raise ValueError('Unknown loss type: {}'.format(cfg.NAME))
