#%%
import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

#%%
# ハードコードされたパラメータ
policy_id = "AriRyo/gray-pickplace-v3_act-policy"  # Hugging Face上のポリシーID（例: AriRyo/pickplace-v3_merged_act-policy）
eval_repo = "AriRyo/val-pickplace-v3_merge"  # Hugging Face上の評価用データセットID
batch_size = 16
num_workers = max(os.cpu_count() // 2, 1)
device = "cuda" if torch.cuda.is_available() else "cpu"

#%%
# 1) 評価データセットをHubから読み込み
ds = LeRobotDataset(eval_repo)  # 変換は評価なので無し
dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#%%
# 2) ポリシー構成と重みをHubから取得し、モデル生成
cfg = PreTrainedConfig.from_pretrained(policy_id)  # ポリシー種別は自動で解決
cfg.pretrained_path = policy_id  # Hub参照を明示
cfg.device = device  # make_policy内でpolicy.to(cfg.device)が呼ばれる
policy = make_policy(cfg, ds_meta=ds.meta).eval()

#%%
# 3) 前後処理（正規化など）をロード
#    原則: 訓練時に保存されたプリプロセッサ/ポストプロセッサを優先
try:
    preproc, postproc = make_pre_post_processors(cfg, pretrained_path=policy_id)
except FileNotFoundError:
    # ない場合は評価データセットの統計で代用
    preproc, postproc = make_pre_post_processors(
        cfg, pretrained_path=policy_id, dataset_stats=ds.meta.stats
    )

#%%
# 4) 推論と指標（アクションMSE）。形状差（チャンク長Tなど）は自動吸収
def align(pred, gt):
    # pred: (B, A) もしくは (B, T, A) を想定。GTは (B, A) が多い。
    if pred.ndim == gt.ndim + 1:  # (B, T, A) vs (B, A)
        # 代表として時刻0を使用（必要なら平均や末尾に変更）
        pred = pred[:, 0]
    if pred.shape[0] != gt.shape[0]:  # 最終バッチなどで長さがずれた場合に揃える
        min_len = min(pred.shape[0], gt.shape[0])
        pred = pred[:min_len]
        gt = gt[:min_len]
    return pred, gt

mse_sum, n_batches = 0.0, 0
with torch.inference_mode():
    iterator = tqdm(dl, desc="evaluating", leave=False)
    for batch in iterator:
        # processorがデバイス配置・正規化を担う（新仕様）
        # 教師アクションはbatch["action"]に入っている想定（v3の基本スキーマ）
        b = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        b_p = preproc(b)  # 観測とアクションの両方が正規化される

        pred_norm = policy.select_action(b_p)  # ポリシー出力（正規化空間）
        pred = postproc(pred_norm)  # 実スケールへ復元
        gt = b["action"]  # 実スケールの教師

        # postproc後はCPUに戻ることがあるためデバイスを揃えてから比較
        pred = pred.to(gt.device)

        pred, gt = align(pred, gt)
        mse = torch.mean((pred - gt) ** 2).item()
        mse_sum += mse
        n_batches += 1
        iterator.set_postfix({"batch_mse": mse})

results = {
    "eval_mse": mse_sum / max(n_batches, 1),
    "n_batches": n_batches,
    "dataset_repo": eval_repo,
    "policy_id": policy_id,
    "device": device,
    "episodes": ds.num_episodes,
}
print("\n===== Evaluation Results =====")
for k, v in results.items():
    print(f"{k:>15}: {v}")
print("============================\n")
