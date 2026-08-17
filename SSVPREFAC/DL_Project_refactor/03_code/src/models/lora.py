"""
Lightweight LoRA utilities for nn.Linear modules.

This module provides a minimal dependency-free LoRA implementation that can be
applied selectively to existing modules.
"""

import math
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wrap a Linear layer with trainable low-rank adapters."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear base module")
        if rank <= 0:
            raise ValueError("rank must be > 0")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))

        # LoRA init: A random, B zeros => no-op at start.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        if freeze_base:
            self.base.weight.requires_grad = False
            if self.base.bias is not None:
                self.base.bias.requires_grad = False

    @property
    def weight(self) -> torch.Tensor:
        # Expose base weight for compatibility with modules that inspect .weight.
        # LoRA contribution is applied in forward to avoid dense weight materialization cost.
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        low_rank = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(low_rank, self.lora_B) * self.scale
        return base_out + lora_out


def apply_lora_to_module(
    module: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    freeze_base: bool = True,
    target_substrings: Optional[Iterable[str]] = None,
    should_apply: Optional[Callable[[str], bool]] = None,
) -> int:
    """
    Recursively replace selected Linear layers with LoRALinear wrappers.

    Args:
        module: root module to edit in-place
        rank/alpha/dropout/freeze_base: LoRA parameters
        target_substrings: if provided, only module names containing any token
            are wrapped. If empty/None, all Linear layers under `module` are used.

    Returns:
        Number of wrapped Linear layers.
    """
    tokens = [t for t in (target_substrings or []) if t]

    def _match(name: str) -> bool:
        if should_apply is not None and not bool(should_apply(name)):
            return False
        if not tokens:
            return True
        lname = name.lower()
        return any(tok.lower() in lname for tok in tokens)

    wrapped = 0

    def _recurse(parent: nn.Module, prefix: str = ""):
        nonlocal wrapped
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and _match(full_name):
                setattr(
                    parent,
                    child_name,
                    LoRALinear(
                        child,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        freeze_base=freeze_base,
                    ),
                )
                wrapped += 1
            else:
                _recurse(child, full_name)

    _recurse(module)
    return wrapped
