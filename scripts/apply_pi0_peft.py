#!/usr/bin/env python
"""Insert OpenPI-style LoRA adapters into Pi0/Pi05 policies without touching `src/`.

The script loads a pretrained policy, injects PEFT adapters into the Paligemma and Gemma
submodules, freezes all non-LoRA parameters (matching `Pi0Config.get_freeze_filter`), and
saves the freshly initialised adapters to disk so that downstream fine-tuning jobs can pick
them up.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from peft import get_peft_model

from scripts.pi0_lora_spec import list_lora_target_modules
from scripts.pi0_lora_utils import (
    POLICY_TYPES,
    ensure_supported_variants,
    freeze_non_lora,
    lazy_import_policy,
    leaf_names,
    make_lora_config,
    write_metadata,
    zero_lora_params,
)


def _save_adapter(peft_module, out_dir: Path, tag: str) -> Path:
    dest = out_dir / tag
    dest.mkdir(parents=True, exist_ok=True)
    peft_module.save_pretrained(dest)
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="HF repo ID or local directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to store adapters")
    parser.add_argument(
        "--policy-type",
        choices=POLICY_TYPES,
        default="pi0",
        help="Choose between Pi0 and Pi05 policies",
    )
    parser.add_argument(
        "--merge-lora",
        action="store_true",
        help="After saving adapters, merge them into the base model and export the merged weights.",
    )
    parser.add_argument(
        "--save-merged-path",
        type=Path,
        default=None,
        help="Optional path to store the merged policy (defaults to <output-dir>/merged_policy)",
    )
    args = parser.parse_args(argv)

    PolicyClass = lazy_import_policy(args.policy_type)
    policy = PolicyClass.from_pretrained(args.base_model)

    pal_spec, gemma_spec = ensure_supported_variants(policy)

    targets = list_lora_target_modules(policy)
    pal_leaf_targets = leaf_names(targets.paligemma)
    gemma_leaf_targets = leaf_names(targets.gemma_expert)

    pal_cfg = make_lora_config(pal_spec.rank, pal_spec.alpha, pal_leaf_targets)
    gemma_cfg = make_lora_config(gemma_spec.rank, gemma_spec.alpha, gemma_leaf_targets)

    pal_module = policy.model.paligemma_with_expert.paligemma
    gemma_module = policy.model.paligemma_with_expert.gemma_expert

    peft_pal = get_peft_model(pal_module, pal_cfg)
    peft_gemma = get_peft_model(gemma_module, gemma_cfg)

    zero_lora_params(peft_pal)
    zero_lora_params(peft_gemma)

    freeze_non_lora(policy)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pal_dir = _save_adapter(peft_pal, args.output_dir, "paligemma")
    gemma_dir = _save_adapter(peft_gemma, args.output_dir, "gemma_expert")
    write_metadata(args.output_dir, policy, pal_spec, gemma_spec, pal_leaf_targets, gemma_leaf_targets)

    if args.merge_lora:
        peft_pal.merge_and_unload()
        peft_gemma.merge_and_unload()
        merged_path = args.save_merged_path or (args.output_dir / "merged_policy")
        merged_path.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(merged_path)

    print(f"Saved paligemma adapter to {pal_dir}")
    print(f"Saved gemma expert adapter to {gemma_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
