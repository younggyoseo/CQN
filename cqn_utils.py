import torch


def random_action_if_within_delta(qs, delta=0.0001):
    q_diff = qs.max(-1).values - qs.min(-1).values
    random_action_mask = q_diff < delta
    if random_action_mask.sum() == 0:
        return None
    argmax_q = qs.max(-1)[1]
    random_actions = torch.randint(0, qs.size(-1), random_action_mask.shape).to(
        qs.device
    )
    argmax_q = torch.where(random_action_mask, random_actions, argmax_q)
    return argmax_q


def encode_action(
    continuous_action: torch.Tensor,
    initial_low: torch.Tensor,
    initial_high: torch.Tensor,
    levels: int,
    bins: int,
):
    """Encode continuous action to discrete action

    Args:
        continuous_action: [..., D] shape tensor
        initial_low: [D] shape tensor consisting of -1
        initial_high: [D] shape tensor consisting of 1
    Returns:
        discrete_action: [..., L, D] shape tensor where L is the level
    """
    low = initial_low.repeat(*continuous_action.shape[:-1], 1)
    high = initial_high.repeat(*continuous_action.shape[:-1], 1)

    idxs = []
    for _ in range(levels):
        # Put continuous values into bin
        slice_range = (high - low) / bins
        idx = torch.floor((continuous_action - low) / slice_range)
        idx = torch.clip(idx, 0, bins - 1)
        idxs.append(idx)

        # Re-compute low/high for each bin (i.e., Zoom-in)
        recalculated_action = low + slice_range * idx
        recalculated_action = torch.clip(recalculated_action, -1.0, 1.0)
        low = recalculated_action
        high = recalculated_action + slice_range
        low = torch.maximum(-torch.ones_like(low), low)
        high = torch.minimum(torch.ones_like(high), high)
    discrete_action = torch.stack(idxs, -2)
    return discrete_action


def decode_action(
    discrete_action: torch.Tensor,
    initial_low: torch.Tensor,
    initial_high: torch.Tensor,
    levels: int,
    bins: int,
):
    """Decode discrete action to continuous action

    Args:
        discrete_action: [..., L, D] shape tensor
        initial_low: [D] shape tensor consisting of -1
        initial_high: [D] shape tensor consisting of 1
    Returns:
        continuous_action: [..., D] shape tensor
    """
    low = initial_low.repeat(*discrete_action.shape[:-2], 1)
    high = initial_high.repeat(*discrete_action.shape[:-2], 1)
    for i in range(levels):
        slice_range = (high - low) / bins
        continuous_action = low + slice_range * discrete_action[..., i, :]
        low = continuous_action
        high = continuous_action + slice_range
        low = torch.maximum(-torch.ones_like(low), low)
        high = torch.minimum(torch.ones_like(high), high)
    continuous_action = (high + low) / 2.0
    return continuous_action


def zoom_in(low: torch.Tensor, high: torch.Tensor, argmax_q: torch.Tensor, bins: int):
    """Zoom-in to the selected interval

    Args:
        low: [D] shape tensor that denotes minimum of the current interval
        high: [D] shape tensor that denotes maximum of the current interval
    Returns:
        low: [D] shape tensor that denotes minimum of the *next* interval
        high: [D] shape tensor that denotes maximum of the *next* interval
    """
    slice_range = (high - low) / bins
    continuous_action = low + slice_range * argmax_q
    low = continuous_action
    high = continuous_action + slice_range
    low = torch.maximum(-torch.ones_like(low), low)
    high = torch.minimum(torch.ones_like(high), high)
    return low, high
