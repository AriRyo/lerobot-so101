#!/usr/bin/env python
"""Utilities describing how OpenPI-style LoRA adapters map onto LeRobot Pi0/Pi05 policies.

The OpenPI reference implementation (see `openpi/models/gemma.py` and `openpi/models/lora.py`)
exposes two LoRA-enabled Gemma variants:

* `gemma_2b_lora` — rank 16 / alpha 16, applied to every Q/K/V/O projection and gated FFN
  matmul in the Paligemma language model.
* `gemma_300m_lora` — rank 32 / alpha 32, applied to the same set of modules inside the
  Gemma action expert.

During fine-tuning, OpenPI freezes every parameter except those created by LoRA adapters
(see `openpi/models/pi0_config.py::Pi0Config.get_freeze_filter`).  This module keeps the same
behavior for the PyTorch ports that live under `lerobot.policies.pi0` and `lerobot.policies.pi05`.
All helpers live outside `src/` so that we can evolve adapter workflows without touching the
core policy implementations.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterable, Literal, Sequence

import torch.nn as nn

__all__ = [
    "LORA_TARGET_LEAVES",
    "PALIGEMMA_LAYERS_PREFIX",
    "GEMMA_EXPERT_LAYERS_PREFIX",
    "LoRAVariantSpec",
    "VARIANT_SPECS",
    "ComponentTargetMap",
    "list_lora_target_modules",
    "write_target_report",
]

# Module leaf names that receive LoRA adapters inside Gemma/PaliGemma layers.
LORA_TARGET_LEAVES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

PALIGEMMA_LAYERS_PREFIX = "model.paligemma_with_expert.paligemma.model.language_model.layers"
GEMMA_EXPERT_LAYERS_PREFIX = "model.paligemma_with_expert.gemma_expert.model.layers"


@dataclasses.dataclass(frozen=True)
class LoRAVariantSpec:
    """LoRA hyper-parameters for a Gemma variant."""

    name: str
    rank: int
    alpha: float
    # We keep target leaves explicit so variants like gemma-13b_lora can tweak coverage later on.
    target_leaves: tuple[str, ...] = LORA_TARGET_LEAVES


# Specs pulled from openpi/models/gemma.py::get_config
VARIANT_SPECS = {
    "gemma_2b":      LoRAVariantSpec(name="gemma_2b",      rank=16, alpha=16.0),
    "gemma_300m":    LoRAVariantSpec(name="gemma_300m",    rank=32, alpha=32.0),
    "gemma_2b_lora": LoRAVariantSpec(name="gemma_2b_lora", rank=16, alpha=16.0),
    "gemma_300m_lora": LoRAVariantSpec(name="gemma_300m_lora", rank=32, alpha=32.0),
}


@dataclasses.dataclass(frozen=True)
class ComponentTargetMap:
    """Container describing LoRA-able module names for both Pi0/Pi05 components."""

    paligemma: tuple[str, ...]
    gemma_expert: tuple[str, ...]

    def for_component(self, component: Literal["paligemma", "gemma_expert"]) -> tuple[str, ...]:
        return getattr(self, component)


def _require_attr(obj, attr: str):
    if not hasattr(obj, attr):  # pragma: no cover - defensive guard
        raise AttributeError(f"Object of type {type(obj)} is missing attribute '{attr}'")
    return getattr(obj, attr)


def list_lora_target_modules(policy: nn.Module) -> ComponentTargetMap:
    """Inspect a PI0/Pi05 policy and list LoRA-able module names.

    Args:
        policy: Instance of `lerobot.policies.pi0.modeling_pi0.PI0Policy` or the Pi05 counterpart.

    Returns:
        ComponentTargetMap where each entry contains fully qualified names that appear inside the
        policy's `named_modules()` traversal.  The lists are sorted to guarantee deterministic output.
    """

    _require_attr(policy, "model")
    _require_attr(policy.model, "paligemma_with_expert")
    _require_attr(policy.model.paligemma_with_expert, "paligemma")
    _require_attr(policy.model.paligemma_with_expert, "gemma_expert")

    named = {name: module for name, module in policy.named_modules() if name}

    def _filter(prefix: str) -> tuple[str, ...]:
        matches: list[str] = []
        for name in named:
            if not name.startswith(prefix):
                continue
            leaf = name.split(".")[-1]
            if leaf in LORA_TARGET_LEAVES:
                matches.append(name)
        return tuple(sorted(matches))

    pal_targets = _filter(PALIGEMMA_LAYERS_PREFIX)
    gemma_targets = _filter(GEMMA_EXPERT_LAYERS_PREFIX)

    return ComponentTargetMap(tuple(sorted(pal_targets)), tuple(sorted(gemma_targets)))


def write_target_report(policy: nn.Module, output: Path) -> None:
    """Utility CLI helper that dumps module targets to disk in plain text."""

    targets = list_lora_target_modules(policy)
    lines: list[str] = ["[paligemma]"]
    lines.extend(targets.paligemma)
    lines.append("")
    lines.append("[gemma_expert]")
    lines.extend(targets.gemma_expert)
    output.write_text("\n".join(lines))


def _main(argv: Sequence[str] | None = None) -> None:
    """Minimal CLI: load a policy and print the target table."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pretrained_path", help="HF repo ID or local directory for the base policy")
    parser.add_argument(
        "--policy",
        choices=("pi0", "pi05"),
        default="pi0",
        help="Policy type to instantiate",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output file for the report")
    args = parser.parse_args(argv)

    if args.policy == "pi0":
        from lerobot.policies.pi0 import PI0Policy as PolicyClass
    else:
        from lerobot.policies.pi05 import PI05Policy as PolicyClass

    policy = PolicyClass.from_pretrained(args.pretrained_path)
    targets = list_lora_target_modules(policy)

    report_lines = [
        f"paligemma targets ({len(targets.paligemma)} modules)",
        *targets.paligemma,
        "",
        f"gemma_expert targets ({len(targets.gemma_expert)} modules)",
        *targets.gemma_expert,
    ]

    report = "\n".join(report_lines)
    print(report)

    if args.output is not None:
        args.output.write_text(report)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _main()
