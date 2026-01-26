import torch


def mse_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.mean((x_hat - x) ** 2)


def l1_loss(h: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(h))


def l0_metric(h: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return (torch.abs(h) > eps).float().sum(dim=1).mean()


def r2_score(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=0, keepdim=True)
    ss_res = torch.sum((x - x_hat) ** 2)
    ss_tot = torch.sum((x - x_mean) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)
