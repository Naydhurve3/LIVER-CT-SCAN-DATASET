import torch


def create_scheduler(optimizer, name="cosine_annealing", **kwargs):
    if name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=kwargs.get("T_max", 50)
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=kwargs.get("step_size", 10), gamma=kwargs.get("gamma", 0.1)
        )
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=kwargs.get("patience", 5)
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")
