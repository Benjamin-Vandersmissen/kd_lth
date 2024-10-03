import torch
import numpy as np

def importance(inp, typ):
    if typ == 'random':
        return torch.rand(inp.shape) + 0.1 # Do this +0.1, to not have 0-values, which might throw off the thresholding.
    else:
        return torch.abs(inp)
        

class PrunableModule(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, to_ignore=None):
        super().__init__()
        if to_ignore is None:
            to_ignore = []
            
        self.model = model
        self.prunable_layers = [name for name, layer in self.model.named_modules()
                                if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear) and name not in to_ignore]
        self.mask = {name: torch.ones_like(layer.weight, dtype=torch.bool, device=layer.weight.device) for name, layer in self.model.named_modules()
                     if name in self.prunable_layers}

    def forward(self, x):
        self.apply_mask()
        return self.model.forward(x)

    def apply_mask(self):
        with torch.no_grad():
            for name in self.prunable_layers:
                layer = self.model.get_submodule(name)
                layer.weight.data *= self.mask[name].to(layer.weight.device)  # Use .data to avoid modifying the computational graph!

    def prune(self, ratio, typ):
        importances = []
        
        for name in self.prunable_layers:
            layer = self.model.get_submodule(name)
            importances.append(importance(layer.weight[self.mask[name]].reshape(-1).cpu(), typ))
        threshold = np.quantile(torch.concat(importances).detach().cpu().numpy(), ratio)  # Needed for large networks, otherwise torch.quantile will throw 'RuntimeError: quantile() input tensor is too large'
        threshold = torch.as_tensor(threshold)
        for name, imp in zip(self.prunable_layers, importances):
            # Both the mask should be 1 (the connection still exists) and the connection should be above threshold
            weight = self.model.get_submodule(name).weight
            self.mask[name][self.mask[name] == 1] = (imp > threshold).to(self.mask[name].device)

    def mask_distance(self, other):
        dis = 0
        for layer in self.prunable_layers:
            dis += torch.logical_xor(self.mask[layer], other.mask[layer]).sum()
        return dis