#!/usr/bin/env python
"""Shared helpers for LoRA/PEFT workflows around Pi0/Pi05 policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import torch.nn as nn
from peft import LoraConfig

from scripts.pi0_lora_spec import LORA_TARGET_LEAVES, VARIANT_SPECS

POLICY_TYPES = {
    "pi0": ("lerobot.policies.pi0", "PI0Policy"),
    "pi05": ("lerobot.policies.pi05", "PI05Policy"),
}


def lazy_import_policy(policy_type: str):
    module_name, class_name = POLICY_TYPES[policy_type]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def ensure_supported_variants(policy) -> Tuple:
    pal_spec = VARIANT_SPECS.get(policy.config.paligemma_variant)
    gemma_spec = VARIANT_SPECS.get(policy.config.action_expert_variant)
    if pal_spec is None or gemma_spec is None:
        raise ValueError(
            "LoRA variant not recognised. Ensure the config variants use *_lora weights as defined in openpi."
        )
    return pal_spec, gemma_spec


def leaf_names(qualified: Iterable[str]) -> list[str]:
    leaves = {name.split(".")[-1] for name in qualified}
    missing = set(LORA_TARGET_LEAVES) - leaves
    if missing:
        raise RuntimeError(f"LoRA target discovery missing leaves: {sorted(missing)}")
    return sorted(leaves)


def make_lora_config(rank: int, alpha: float, targets: list[str]) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=targets,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )


def zero_lora_params(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        if "lora_" in name:
            nn.init.zeros_(param)


def freeze_non_lora(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        param.requires_grad = "lora_" in name


def write_metadata(out_dir: Path, policy, pal_spec, gemma_spec, pal_targets, gemma_targets) -> None:
    meta = {
        "policy_type": policy.name,
        "paligemma_variant": policy.config.paligemma_variant,
        "action_expert_variant": policy.config.action_expert_variant,
        "paligemma_spec": {
            "rank": pal_spec.rank,
            "alpha": pal_spec.alpha,
            "target_modules": pal_targets,
        },
        "gemma_spec": {
            "rank": gemma_spec.rank,
            "alpha": gemma_spec.alpha,
            "target_modules": gemma_targets,
        },
    }
    (out_dir / "adapter_manifest.json").write_text(json.dumps(meta, indent=2))
