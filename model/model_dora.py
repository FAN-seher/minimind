import torch
from torch import nn


class LoRAAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


class DoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0, init_m=None):
        super().__init__()
        self.alpha = alpha
        self.m = nn.Parameter(init_m if init_m is not None else torch.ones(in_features))
        self.adapter = LoRAAdapter(in_features, out_features, rank)

    def forward(self, x):
        x_scaled = x * self.m
        return self.alpha * self.adapter(x_scaled)


def apply_dora(model, rank=16, alpha=1.0):
    eps = 1e-6
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            original_weight = module.weight.data.clone()
            column_norm = original_weight.norm(dim=0, keepdim=True)
            direction = original_weight / (column_norm + eps)
            module.weight.data = direction
            module.weight.requires_grad = False

            dora = DoRA(module.weight.shape[1], module.weight.shape[0], rank=rank, alpha=alpha, init_m=column_norm.squeeze(0)).to(module.weight.device)
            setattr(module, 'dora', dora)
            original_forward = module.forward

            def forward_with_dora(x, layer1=original_forward, layer2=dora):
                x_scaled = x * layer2.m
                return layer1(x_scaled) + layer2(x_scaled)

            module.forward = forward_with_dora


def load_dora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'dora'):
            dora_state = {k.replace(f'{name}.dora.', ''): v for k, v in state_dict.items() if f'{name}.dora.' in k}
            module.dora.load_state_dict(dora_state)


def save_dora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'dora'):
            clean_name = name[7:] if name.startswith('module.') else name
            dora_state = {f'{clean_name}.dora.{k}': v.cpu().half() for k, v in module.dora.state_dict().items()}
            state_dict.update(dora_state)
    torch.save(state_dict, path)


def merge_dora(model, dora_path, save_path):
    load_dora(model, dora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.dora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.dora.' not in name:
            base_direction = module.weight.data.clone()
            if hasattr(module, 'dora'):
                adapter = module.dora.adapter
                combined = base_direction + (adapter.B.weight.data @ adapter.A.weight.data)
                full_weight = combined * module.dora.m.unsqueeze(0)
                state_dict[f'{name}.weight'] = full_weight.cpu().half()
            else:
                state_dict[f'{name}.weight'] = base_direction.cpu().half()
    torch.save(state_dict, save_path)
