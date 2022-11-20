import torch


def summeary_torch_model_parameters(model):
    print("model parameter summary:")
    for name, param in model.named_parameters():
        param_mean = torch.mean(param)
        param_std = torch.std(param)
        param_max = torch.max(param)
        param_min = torch.min(param)
        print(f"\t{name}:\t\t\t{param_mean:.4e} +- {param_std:.4e}. max={param_max:.4e} min={param_min:.4e}")
