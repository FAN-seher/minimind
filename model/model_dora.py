import torch
from torch import nn
import torch.nn.functional as F


class DoRA(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.eps = eps

        # 预训练权重
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_dropout = nn.Identity()

        self.magnitude = nn.Parameter(torch.ones(1, in_features))

        self.reset_parameters()

        # 冻结 base weight
        self.weight.requires_grad = False

    def reset_parameters(self):

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)

        with torch.no_grad():

            col_norm = torch.norm(self.weight, dim=0, keepdim=True)
            col_norm = torch.clamp(col_norm, min=self.eps)

            self.magnitude.copy_(col_norm)

    def _effective_weight(self):

        W0 = self.weight

        if self.r > 0:

            delta_V = self.scaling * (self.lora_B @ self.lora_A)

            V = W0 + delta_V

        else:

            V = W0

        col_norm = torch.norm(V, dim=0, keepdim=True)
        col_norm = torch.clamp(col_norm, min=self.eps)

        direction = V / col_norm

        W_eff = direction * self.magnitude

        return W_eff

    def forward(self, x):

        W_eff = self._effective_weight()

        return F.linear(x, W_eff, self.bias)

def apply_dora(model, rank=16):

    for name, module in model.named_modules():

        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:

            dora = DoRA(
                module.in_features,
                module.out_features,
                r=rank,
                bias=(module.bias is not None)
            ).to(module.weight.device)

            # 复制原始权重
            dora.weight.data.copy_(module.weight.data)

            if module.bias is not None:
                dora.bias.data.copy_(module.bias.data)

            setattr(module, "dora", dora)

            # forward替换
            def forward_with_dora(x, layer2=dora):
                return layer2(x)

            module.forward = forward_with_dora

def load_dora(model, path):

    state_dict = torch.load(path, map_location=model.device)

    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():

        if hasattr(module, 'dora'):

            dora_state = {
                k.replace(f'{name}.dora.', ''): v
                for k, v in state_dict.items()
                if f'{name}.dora.' in k
            }

            module.dora.load_state_dict(dora_state)

def save_dora(model, path):

    raw_model = getattr(model, '_orig_mod', model)

    state_dict = {}

    for name, module in raw_model.named_modules():

        if hasattr(module, 'dora'):

            clean_name = name[7:] if name.startswith("module.") else name

            dora_state = {
                f'{clean_name}.dora.{k}': v.cpu().half()
                for k, v in module.dora.state_dict().items()
            }

            state_dict.update(dora_state)

    torch.save(state_dict, path)
