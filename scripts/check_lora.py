#!/usr/bin/env python
import argparse
import json
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch

def import_pi05_policy():
    # LeRobot のパス差分に備えて両方試す
    try:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        return PI05Policy
    except Exception:
        from lerobot.common.policies.pi05.modeling_pi05 import PI05Policy
        return PI05Policy

def main(repo_id: str, device: str):
    # 1) Hub から config.json を読む（policy.type の確認用）
    cfg_path = hf_hub_download(repo_id, "config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    policy_type = cfg.get("type", None)
    print(f"[CONFIG] type={policy_type}")

    # 2) Hub から重みを読む（LoRAキーの存在確認）
    w_path = hf_hub_download(repo_id, "model.safetensors")
    sd = load_file(w_path)
    keys = list(sd.keys())
    lora_keys = [k for k in keys if "lora_" in k]
    print(f"[WEIGHT] num_keys={len(keys)}, num_lora_keys={len(lora_keys)}")
    if lora_keys:
        print("[WEIGHT] example_lora_keys:")
        for k in lora_keys[:10]:
            print("  ", k)

    # 3) PI05Policy としてロード（ここで LoRA が自動で組まれないと、lora_ は使われない）
    PI05Policy = import_pi05_policy()
    policy = PI05Policy.from_pretrained(repo_id)
    policy.to(device)
    policy.eval()

    # 4) ロード後のモデル内部に lora_ パラメータが存在するか
    named = list(policy.named_parameters())
    lora_named = [n for (n, p) in named if "lora_" in n]
    print(f"[MODEL] num_named_parameters={len(named)}")
    print(f"[MODEL] num_lora_parameters_in_model={len(lora_named)}")
    if lora_named:
        print("[MODEL] example_lora_parameter_names:")
        for n in lora_named[:10]:
            print("  ", n)

    # 5) state_dict を strict=False で再ロードして「unexpected_keys」を観察
    #    ここで unexpected_keys に lora_ が大量に出るなら、LoRAモジュールが構築されておらず捨てられている
    incompatible = policy.load_state_dict(sd, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    u_lora = [k for k in unexpected if "lora_" in k]
    print(f"[LOAD_CHECK] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
    print(f"[LOAD_CHECK] unexpected_lora_keys={len(u_lora)}")
    if u_lora:
        print("[LOAD_CHECK] example_unexpected_lora_keys:")
        for k in u_lora[:10]:
            print("  ", k)

    # 6) ベースと一致していないか簡易チェック（LoRAを凍結学習していると“ベース部分”は一致しがち）
    #    代表キーをいくつか抜き出して比較したい場合は、pi05_base も同様に sd を読んで差分を見る
    print("[DONE]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    main(args.repo_id, args.device)
