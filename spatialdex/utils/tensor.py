import torch


def move_masked_to_left_ids(tensor, mask, pad_zero=True):
    masked_index = mask.cumsum(dim=1) - 1
    unmasked_index = (~mask).cumsum(dim=1) - 1
    unmasked_index += mask.sum(dim=1).unsqueeze(1)
    s2t_index = torch.where(mask, masked_index, unmasked_index)
    t2s_index = torch.argsort(s2t_index, dim=1)
    result = torch.gather(tensor, 1, t2s_index)
    length = mask.sum(dim=1)
    result = result[:, : length.max()]
    new_mask = torch.arange(result.shape[1], device=length.device).unsqueeze(0) < length.unsqueeze(1)
    if pad_zero:
        result[~new_mask] = 0
    return (result, new_mask)


def move_masked_to_left(tensor, mask, pad_zero=True):
    masked_index = mask.cumsum(dim=1) - 1
    unmasked_index = (~mask).cumsum(dim=1) - 1
    unmasked_index += mask.sum(dim=1).unsqueeze(1)
    s2t_index = torch.where(mask, masked_index, unmasked_index)
    t2s_index = torch.argsort(s2t_index, dim=1)
    result = torch.gather(tensor, 1, t2s_index.unsqueeze(2).expand(-1, -1, tensor.shape[2]))
    length = mask.sum(dim=1)
    result = result[:, : length.max()]
    new_mask = torch.arange(result.shape[1], device=length.device).unsqueeze(0) < length.unsqueeze(1)
    if pad_zero:
        result[~new_mask] = 0
    return (result, new_mask)
