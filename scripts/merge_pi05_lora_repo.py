#!/usr/bin/env python
import argparse
import os
import shutil
import tempfile

from huggingface_hub import snapshot_download, hf_hub_download, HfApi
from safetensors.torch import load_file
import torch

from peft import LoraConfig, get_peft_model

def import_pi05_policy():
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    return PI05Policy

LORA_TARGET_LEAVES = (
    "q_proj","k_proj","v_proj","o_proj",
    "gate_proj","up_proj","down_proj",
)

def inject_pi05_lora(policy):
    pal = policy.model.paligemma_with_expert.paligemma
    gem = policy.model.paligemma_with_expert.gemma_expert

    pal_cfg = LoraConfig(
        r=16, lora_alpha=16, target_modules=list(LORA_TARGET_LEAVES),
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    gem_cfg = LoraConfig(
        r=32, lora_alpha=32, target_modules=list(LORA_TARGET_LEAVES),
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )

    # 重要: 返り値で必ず差し替える
    policy.model.paligemma_with_expert.paligemma = get_peft_model(pal, pal_cfg)
    policy.model.paligemma_with_expert.gemma_expert = get_peft_model(gem, gem_cfg)

    # 注入確認
    lora_params = [n for n, _ in policy.named_parameters() if "lora_" in n]
    assert len(lora_params) > 0, "LoRA注入に失敗しています（lora_が0）"

def main(lora_repo, merged_repo, base_repo="lerobot/pi05_base", device="cpu"):
    # 1) LoRA repo を丸ごとローカルに落とす（processor類を維持するため）
    src_dir = snapshot_download(lora_repo)
    out_dir = tempfile.mkdtemp(prefix="pi05_lora_merged_")

    # repo内容をコピー（あとで model.safetensors だけ差し替える）
    shutil.copytree(src_dir, out_dir, dirs_exist_ok=True)

    # 2) LoRA の重みを読む
    w_path = hf_hub_download(lora_repo, filename="model.safetensors")
    sd = load_file(w_path)
    print("[INFO] lora_keys_in_weight:", sum("lora_" in k for k in sd.keys()))

    # 3) ベースをロードして LoRA 構造を注入
    PI05Policy = import_pi05_policy()
    policy = PI05Policy.from_pretrained(base_repo)
    policy.to(device)

    inject_pi05_lora(policy)

    # 4) LoRA repo の state_dict をロード（LoRA構造があるので base_layer/lora_ を受けられる）
    incompatible = policy.load_state_dict(sd, strict=False)
    print("[INFO] missing:", len(incompatible.missing_keys), "unexpected:", len(incompatible.unexpected_keys))
    # この時点で unexpected_lora が大量なら注入先が違う
    unexpected_lora = [k for k in incompatible.unexpected_keys if "lora_" in k]
    assert len(unexpected_lora) == 0, f"LoRAキーがunexpectedになっています（注入先不一致）: {unexpected_lora[:3]}"

    # 5) マージして LoRA を焼き込む（破壊的なのでこの後は “通常モデル”）
    pal = policy.model.paligemma_with_expert.paligemma
    gem = policy.model.paligemma_with_expert.gemma_expert
    policy.model.paligemma_with_expert.paligemma = pal.merge_and_unload()
    policy.model.paligemma_with_expert.gemma_expert = gem.merge_and_unload()

    # 6) 一時ディレクトリに通常形式で保存し、model.safetensors だけ out_dir に上書き
    with tempfile.TemporaryDirectory(prefix="save_") as tmp_save:
        policy.save_pretrained(tmp_save)
        shutil.copy2(os.path.join(tmp_save, "model.safetensors"),
                     os.path.join(out_dir, "model.safetensors"))
        # config.json も save_pretrained のものに寄せたい場合は下も有効化
        shutil.copy2(os.path.join(tmp_save, "config.json"),
                     os.path.join(out_dir, "config.json"))

    # 7) Hub に push
    api = HfApi()
    api.create_repo(merged_repo, exist_ok=True)
    api.upload_folder(repo_id=merged_repo, folder_path=out_dir)
    print("[DONE] pushed:", merged_repo)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_repo", required=True)
    ap.add_argument("--merged_repo", required=True)
    ap.add_argument("--base_repo", default="lerobot/pi05_base")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    main(args.lora_repo, args.merged_repo, args.base_repo, args.device)

