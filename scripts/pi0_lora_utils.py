#!/usr/bin/env python
"""Shared helpers for LoRA/PEFT workflows around Pi0/Pi05 policies."""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

from scripts.pi0_lora_spec import LORA_TARGET_LEAVES, VARIANT_SPECS, list_lora_target_modules

POLICY_TYPES = {
    "pi0": ("lerobot.policies.pi0", "PI0Policy"),
    "pi05": ("lerobot.policies.pi05", "PI05Policy"),
}


@dataclass(slots=True)
class LoRAHandles:
    """Container bundling PEFT modules and metadata for later export."""

    paligemma: nn.Module
    gemma_expert: nn.Module
    pal_spec: "LoRAVariantSpec"
    gemma_spec: "LoRAVariantSpec"
    pal_targets: list[str]
    gemma_targets: list[str]


def lazy_import_policy(policy_type: str):
    module_name, class_name = POLICY_TYPES[policy_type]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def _maybe_override_spec(spec, *, rank: int | None = None, alpha: float | None = None):
    if rank is None and alpha is None:
        return spec
    updates = {}
    if rank is not None:
        updates["rank"] = rank
    if alpha is not None:
        updates["alpha"] = alpha
    return dataclasses.replace(spec, **updates)  # type: ignore[arg-type]


def attach_lora_adapters(
    policy,
    *,
    rank_override: int | None = None,
    alpha_override: float | None = None,
    zero_init: bool = True,
    init_adapter_dir: Path | None = None,
) -> LoRAHandles:
    """Inject PEFT adapters into paligemma/gemma modules and optionally warm start them."""

    pal_spec, gemma_spec = ensure_supported_variants(policy)
    pal_spec = _maybe_override_spec(pal_spec, rank=rank_override, alpha=alpha_override)
    gemma_spec = _maybe_override_spec(gemma_spec, rank=rank_override, alpha=alpha_override)

    targets = list_lora_target_modules(policy)
    pal_leaf_targets = leaf_names(targets.paligemma)
    gemma_leaf_targets = leaf_names(targets.gemma_expert)

    pal_cfg = make_lora_config(pal_spec.rank, pal_spec.alpha, pal_leaf_targets)
    gemma_cfg = make_lora_config(gemma_spec.rank, gemma_spec.alpha, gemma_leaf_targets)

    pal_module = policy.model.paligemma_with_expert.paligemma
    gemma_module = policy.model.paligemma_with_expert.gemma_expert

    peft_pal = get_peft_model(pal_module, pal_cfg)
    peft_gemma = get_peft_model(gemma_module, gemma_cfg)

    if zero_init:
        zero_lora_params(peft_pal)
        zero_lora_params(peft_gemma)

    if init_adapter_dir is not None:
        _load_adapter_safetensors(peft_pal, init_adapter_dir / "paligemma")
        _load_adapter_safetensors(peft_gemma, init_adapter_dir / "gemma_expert")

    return LoRAHandles(
        paligemma=peft_pal,
        gemma_expert=peft_gemma,
        pal_spec=pal_spec,
        gemma_spec=gemma_spec,
        pal_targets=pal_leaf_targets,
        gemma_targets=gemma_leaf_targets,
    )


def _load_adapter_safetensors(module: nn.Module, adapter_dir: Path) -> bool:
    """Best-effort load of LoRA weights saved via PEFT."""

    if not adapter_dir.exists():
        return False
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.exists():
        return False
    state = load_file(str(weights_path))
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        logging.warning(
            "Adapter load from %s had missing=%s, unexpected=%s", adapter_dir, missing, unexpected
        )
    return True


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


def export_lora_adapters(policy, handles: LoRAHandles, output_dir: Path) -> tuple[Path, Path]:
    """Save LoRA adapters and manifest to ``output_dir``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    pal_dir = output_dir / "paligemma"
    pal_dir.mkdir(parents=True, exist_ok=True)
    handles.paligemma.save_pretrained(pal_dir)

    gemma_dir = output_dir / "gemma_expert"
    gemma_dir.mkdir(parents=True, exist_ok=True)
    handles.gemma_expert.save_pretrained(gemma_dir)

    write_metadata(
        output_dir,
        policy,
        handles.pal_spec,
        handles.gemma_spec,
        handles.pal_targets,
        handles.gemma_targets,
    )
    return pal_dir, gemma_dir
