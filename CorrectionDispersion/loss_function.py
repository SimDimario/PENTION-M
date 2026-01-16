import torch
import torch.nn.functional as F


def total_variation(x):
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw


def physics_masked_loss(
    output, sim, mask, alpha=1.0, beta=10.0, gamma=0.1, delta=1.0, sigma=5.0
):
    """
    output: [B, m, m] -> corrective network predictions
    sim:    [B, m, m] -> raw simulation
    mask:   [m, m]    -> 1 = free space, 0 = building

    alpha: weight for consistency with simulation in free zones
    beta:  weight for penalty on concentration in buildings
    gamma: weight for smoothness regularization
    delta: weight for positivity
    """

    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.float32, device=output.device)
    else:
        mask = mask.to(output.device)

    mse_free = torch.mean(((output - sim) ** 2) * mask)

    building_penalty = torch.mean((output**2) * (1 - mask))

    dx = output[:, 1:, :] - output[:, :-1, :]
    dy = output[:, :, 1:] - output[:, :, :-1]
    smoothness = torch.mean(dx**2) + torch.mean(dy**2)

    negativity = torch.mean(torch.relu(-output) ** 2)

    loss = (
        alpha * mse_free
        + beta * building_penalty
        + gamma * smoothness
        + delta * negativity
    )
    return loss


def physics_informed_loss(output, mc, mask, alpha=1.0, beta=10.0, gamma=0.1):
    """
    output: [B, m, m] -> corrective network predictions
    mc: [B, m, m] -> raw simulation input
    mask: [m, m] -> 1 = free space, 0 = building
    alpha, beta, gamma, delta: weights for various components of the loss
    """

    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.float32, device=output.device)
    else:
        mask = mask.to(output.device)

    mse_free = torch.mean(((output - mc[:, 0]) ** 2) * mask)
    building_penalty = torch.mean((output**2) * (1 - mask))
    dx = output[:, 1:, :] - output[:, :-1, :]
    dy = output[:, :, 1:] - output[:, :, :-1]
    smoothness = torch.mean(dx**2) + torch.mean(dy**2)
    loss = alpha * mse_free + beta * building_penalty + gamma * smoothness

    return loss
