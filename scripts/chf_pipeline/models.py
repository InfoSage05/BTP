"""
Two candidate architectures for Stage 1, trained and evaluated identically
so the comparison is fair. Both operate on the same feature vector
(see data_prep.FEATURE_COLS) and predict log(CHF) (see train_stage1.py for
why: CHF spans ~1-50,000 kW/m^2, log-space keeps the loss well-scaled).

SmallMLP: matches the architecture that won Yang et al. (2025)'s
extrapolation benchmark -- 2 hidden layers, 256/128 nodes. Included as-is,
not as a strawman, because it's the only architecture in this literature
with real, benchmarked extrapolation numbers behind it.

FTTransformer: a from-scratch, compact Feature-Tokenizer Transformer
(Gorishniy et al. 2021 pattern -- each scalar feature gets its own learned
embedding, a [CLS]-style aggregation token attends over all feature
embeddings via self-attention, final prediction from the CLS token). No
CHF-specific pretrained weights exist anywhere (checked in conversation
this project), so this is a genuine from-scratch implementation, not a
downloaded checkpoint.
"""
import torch
import torch.nn as nn


class SmallMLP(nn.Module):
    def __init__(self, n_features: int, hidden=(256, 128), dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FeatureTokenizer(nn.Module):
    """Each of the n_features scalar inputs gets its own learned linear
    embedding into d_model, i.e. token_i = x_i * W_i + b_i."""
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))

    def forward(self, x):
        # x: (batch, n_features) -> tokens: (batch, n_features, d_model)
        return x.unsqueeze(-1) * self.weight + self.bias


class FTTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        tokens = self.tokenizer(x)                       # (B, F, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)    # (B, 1, D)
        seq = torch.cat([cls, tokens], dim=1)             # (B, F+1, D)
        encoded = self.encoder(seq)
        return self.head(encoded[:, 0]).squeeze(-1)       # prediction from CLS token


class LoRALinear(nn.Module):
    """A frozen base nn.Linear plus a trainable low-rank delta: y = W0*x + b0
    + (B @ A) * x * scale. Only A, B are trainable; W0/b0 are frozen copies
    of a pretrained layer's weights."""
    def __init__(self, base_linear: nn.Linear, rank: int = 4, scale: float = 1.0):
        super().__init__()
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.weight = base_linear.weight.detach().clone()
        self.bias = base_linear.bias.detach().clone() if base_linear.bias is not None else None
        self.A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        self.scale = scale

    def forward(self, x):
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        delta = torch.nn.functional.linear(torch.nn.functional.linear(x, self.A), self.B)
        return base + self.scale * delta


class LoRAMLP(nn.Module):
    """Wraps a pretrained SmallMLP: every Linear becomes a frozen-base +
    trainable-low-rank-delta LoRALinear. Only LoRA params (A, B matrices)
    have requires_grad=True -- the base MLP's original weights are frozen
    copies, never updated."""
    def __init__(self, base_mlp: "SmallMLP", rank: int = 4):
        super().__init__()
        layers = []
        for layer in base_mlp.net:
            if isinstance(layer, nn.Linear):
                layers.append(LoRALinear(layer, rank=rank))
            else:
                layers.append(layer)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def lora_parameters(self):
        for m in self.modules():
            if isinstance(m, LoRALinear):
                yield m.A
                yield m.B


class MoEGate(nn.Module):
    """Small 2-expert soft gate over the raw input features -- a 2-layer
    MLP producing a softmax weight for (flow_expert, pool_expert)."""
    def __init__(self, n_features: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)  # (batch, 2)


class MoEModel(nn.Module):
    """Combines a frozen flow-boiling expert and a trainable pool-boiling
    expert via a learned softmax gate over the same input features. Both
    experts see the same (P,G,X,D,...) feature vector -- G=0 rows are pool
    boiling, G>0 rows are flow boiling, and the gate is expected to learn
    to route accordingly, though it is never told this rule directly."""
    def __init__(self, flow_expert: nn.Module, pool_expert: nn.Module, n_features: int):
        super().__init__()
        self.flow_expert = flow_expert
        for p in self.flow_expert.parameters():
            p.requires_grad = False
        self.pool_expert = pool_expert
        self.gate = MoEGate(n_features)

    def forward(self, x):
        w = self.gate(x)  # (batch, 2)
        flow_pred = self.flow_expert(x)
        pool_pred = self.pool_expert(x)
        return w[:, 0] * flow_pred + w[:, 1] * pool_pred

    def forward_with_gate(self, x):
        w = self.gate(x)
        flow_pred = self.flow_expert(x)
        pool_pred = self.pool_expert(x)
        return w[:, 0] * flow_pred + w[:, 1] * pool_pred, w


class LoRAFTTransformer(nn.Module):
    """LoRA-style adapter for FTTransformer: the pretrained self-attention
    encoder is frozen entirely (that's the expensive, general-purpose part).
    Trainable: a low-rank delta on the feature tokenizer's per-feature
    embedding weights (A,B rank-decomposition of the (n_features, d_model)
    weight matrix) plus the small prediction head (already <5k params,
    counts as an adapter given the frozen encoder is the bulk of the model).
    This is a scoped-down LoRA relative to LoRAMLP (which wraps every
    Linear) -- reaching into every internal nn.TransformerEncoderLayer
    Linear (in_proj/out_proj/linear1/linear2) for full LoRA coverage was
    out of scope for this pass; freezing the encoder wholesale and adapting
    the tokenizer + head is a standard, defensible reduced form.
    """
    def __init__(self, base: "FTTransformer", rank: int = 4):
        super().__init__()
        self.tokenizer_weight = base.tokenizer.weight.detach().clone()
        self.tokenizer_bias = base.tokenizer.bias.detach().clone()
        n_features, d_model = self.tokenizer_weight.shape
        self.A = nn.Parameter(torch.randn(rank, n_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_model, rank))

        self.cls_token = base.cls_token.detach().clone()
        self.encoder = base.encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        import copy
        self.head = copy.deepcopy(base.head)  # trainable, small

    def forward(self, x):
        delta = torch.einsum('bf,rf->br', x, self.A)
        delta = torch.einsum('br,dr->bd', delta, self.B)  # (batch, d_model)
        tokens = x.unsqueeze(-1) * self.tokenizer_weight + self.tokenizer_bias
        tokens = tokens + delta.unsqueeze(1)  # broadcast delta onto every feature token
        cls = self.cls_token.expand(x.size(0), -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(seq)
        return self.head(encoded[:, 0]).squeeze(-1)

    def lora_parameters(self):
        yield self.A
        yield self.B
        for p in self.head.parameters():
            yield p
