import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch.utils.checkpoint import checkpoint


@dataclass
class GPTConfig:
    vocab_size: int = 32768
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 384
    intermediate_size: Optional[int] = None
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        if self.intermediate_size is None:
            hidden_size = int(8 * self.n_embd / 3)
            self.intermediate_size = 256 * ((hidden_size + 255) // 256)

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()

        inv_freq = 1.0 / (
            base ** (
                torch.arange(
                    0,
                    dim,
                    2,
                    dtype=torch.float32,
                )
                / dim
            )
        )

        positions = torch.arange(
            max_position_embeddings,
            dtype=torch.float32,
        )

        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer(
            "cos_cached",
            emb.cos(),
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            emb.sin(),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        cos = self.cos_cached[:seq_len].to(
            device=x.device,
            dtype=x.dtype,
        )

        sin = self.sin_cached[:seq_len].to(
            device=x.device,
            dtype=x.dtype,
        )

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    cos = rearrange(
        cos,
        "t d -> 1 1 t d",
    )

    sin = rearrange(
        sin,
        "t d -> 1 1 t d",
    )

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)

    return q, k


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        intermediate_size = config.intermediate_size

        if intermediate_size is None:
            raise ValueError("intermediate_size must be set")

        self.gate_proj = nn.Linear(
            config.n_embd,
            intermediate_size,
            bias=config.bias,
        )

        self.up_proj = nn.Linear(
            config.n_embd,
            intermediate_size,
            bias=config.bias,
        )

        self.down_proj = nn.Linear(
            intermediate_size,
            config.n_embd,
            bias=config.bias,
        )

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.dropout = config.dropout

        self.q_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
        )

        self.k_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
        )

        self.v_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
        )

        self.out_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
        )

        self.resid_dropout = nn.Dropout(config.dropout)

        self.rope = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.block_size,
            base=config.rope_theta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape

        q = rearrange(
            self.q_proj(x),
            "b t (h d) -> b h t d",
            h=self.n_head,
        )

        k = rearrange(
            self.k_proj(x),
            "b t (h d) -> b h t d",
            h=self.n_head,
        )

        v = rearrange(
            self.v_proj(x),
            "b t (h d) -> b h t d",
            h=self.n_head,
        )

        cos, sin = self.rope(
            x,
            seq_len,
        )

        q, k = apply_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
        )

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = rearrange(
            y,
            "b h t d -> b t (h d)",
        )

        y = self.out_proj(y)

        return self.resid_dropout(y)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.input_layernorm = nn.RMSNorm(
            config.n_embd
        )

        self.self_attn = CausalSelfAttention(
            config
        )

        self.post_attention_layernorm = nn.RMSNorm(
            config.n_embd
        )

        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(
            self.input_layernorm(x)
        )

        x = x + self.mlp(
            self.post_attention_layernorm(x)
        )

        return x


class TransformerModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.n_embd,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layer)
            ]
        )

        self.norm = nn.RMSNorm(
            config.n_embd
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        gradient_checkpointing: bool = False,
    ) -> torch.Tensor:

        x = self.embed_tokens(input_ids)

        for layer in self.layers:

            if gradient_checkpointing and self.training:
                x = checkpoint(
                    layer,
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(x)

        return self.norm(x)


class LinuxSLM(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.model = TransformerModel(config)

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        self.gradient_checkpointing = False
        self.config = config

        self.apply(self._init_weights)

        # Weight tying
        self.lm_head.weight = self.model.embed_tokens.weight

        self._init_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def _init_residual_projections(self) -> None:

        std = 0.02 / math.sqrt(
            2 * self.config.n_layer
        )

        for name, module in self.named_modules():

            if name.endswith(
                ("out_proj", "down_proj")
            ) and isinstance(module, nn.Linear):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=std,
                )

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        if input_ids.size(1) > self.config.block_size:
            raise ValueError(
                "input sequence length exceeds block_size"
            )

        hidden_states = self.model(
            input_ids,
            self.gradient_checkpointing,
        )

        logits = self.lm_head(hidden_states)

        loss = None

        if labels is not None:

            # Causal language modeling:
            #
            # get_batch() already constructs:
            #   x[t] = token[t]
            #   y[t] = token[t+1]
            #
            # So logits[t] should predict labels[t] directly (no additional shift).

            loss = F.cross_entropy(
                rearrange(logits, "b t v -> (b t) v"),
                rearrange(labels, "b t -> (b t)"),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:

        was_training = self.training

        self.eval()

        for _ in range(max_new_tokens):

            idx_cond = input_ids[
                :,
                -self.config.block_size:
            ]

            logits, _ = self(idx_cond)

            # Last position predicts the next token.
            logits = logits[:, -1, :]

            if temperature <= 0.0:

                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True,
                )

            else:

                logits = logits / temperature

                if top_k is not None:

                    k = min(
                        top_k,
                        logits.size(-1),
                    )

                    values, _ = torch.topk(
                        logits,
                        k,
                    )

                    logits = logits.masked_fill(
                        logits < values[:, [-1]],
                        -float("inf"),
                    )

                if do_sample:

                    probs = F.softmax(
                        logits,
                        dim=-1,
                    )

                    next_token = torch.multinomial(
                        probs,
                        num_samples=1,
                    )

                else:

                    next_token = torch.argmax(
                        logits,
                        dim=-1,
                        keepdim=True,
                    )

            input_ids = torch.cat(
                (input_ids, next_token),
                dim=1,
            )

        if was_training:
            self.train()

        return input_ids

    def configure_optimizer(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:

        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():

            if not param.requires_grad:
                continue

            if (
                param.dim() >= 2
                and not name.endswith(
                    "embed_tokens.weight"
                )
            ):
                decay_params.append(param)

            else:
                no_decay_params.append(param)

        optimizer_groups = [
            {
                "params": decay_params,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

        fused_available = (
            "fused"
            in torch.optim.AdamW.__init__.__code__.co_varnames
        )

        if fused_available and device_type == "cuda":

            return torch.optim.AdamW(
                optimizer_groups,
                lr=learning_rate,
                betas=betas,
                fused=True,
            )

        return torch.optim.AdamW(
            optimizer_groups,
            lr=learning_rate,
            betas=betas,
        )

    def num_parameters(
        self,
        non_embedding: bool = True,
    ) -> int:

        total = sum(
            param.numel()
            for param in self.parameters()
        )

        if non_embedding:
            total -= (
                self.model.embed_tokens.weight.numel()
            )

        return total

    def estimate_mfu(
        self,
        fwdbwd_per_iter: int,
        dt: float,
        flops_promised: float = 65e12,
    ) -> float:

        n_params = self.num_parameters(
            non_embedding=False
        )

        cfg = self.config

        flops_per_token = (
            6 * n_params
            + 12
            * cfg.n_layer
            * cfg.n_head
            * cfg.head_dim
            * cfg.block_size
        )

        flops_per_iter = (
            flops_per_token
            * cfg.block_size
            * fwdbwd_per_iter
        )

        flops_achieved = (
            flops_per_iter / dt
        )

        return (
            flops_achieved
            / flops_promised
        )