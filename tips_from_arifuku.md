
テレオペのコマンド



lerobot-teleoperate --robot.type=so101_follower   --robot.port=/dev/ttyACM0  --robot.id=my_follower_arm --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_leader_arm



データ収集のコマンド


lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm \
    --robot.cameras="{ above: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --display_data=true \
    --dataset.repo_id=AriRyo/gray-pickplace-v3 \
    --dataset.num_episodes=28 \
    --dataset.single_task="Pick the gray cube and place it on the circle." \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=5 \
    --dataset.push_to_hub=true \
    --resume=true


赤黒のデータ収集コマンド

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm \
    --robot.cameras="{ above: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --display_data=true \
    --dataset.repo_id=AriRyo/red-pickplace-v3 \
    --dataset.num_episodes=56 \
    --dataset.single_task="Pick the red cube and place it on the circle." \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=5 \
    --dataset.push_to_hub=true \
    --resume=true


データセットのマージコマンド
lerobot-edit-dataset \
    --repo_id AriRyo/pickplace-v3_merged \
    --operation.type merge \
    --operation.repo_ids "['AriRyo/gray-pickplace-v3', 'AriRyo/redblack-pickplace-v3']" \
    --push_to_hub true

注意: LeRobot 0.4.1のバグ修正として、src/lerobot/datasets/dataset_tools.pyの259行目で
video_files_size_in_mb=5000を設定しています（デフォルトは500MB）。
これにより、ビデオが複数ファイルに分割されることを防ぎます。
詳細: https://github.com/huggingface/lerobot/issues/2328


学習コマンド
ACT
gray
nohup lerobot-train   --dataset.repo_id=AriRyo/gray-pickplace-v3   --policy.type=act   --output_dir=outputs/train/act_gray-pickplace-v3   --job_name=act_gray-pickplace-v3   --policy.device=cuda   --wandb.enable=true   --policy.repo_id=AriRyo/gray-pickplace-v3_act-policy > output.log &

redblack
lerobot-train   --dataset.repo_id=AriRyo/redblack-pickplace-v3   --policy.type=act   --output_dir=outputs/train/act_redblack-pickplace-v3   --job_name=act_redblack-pickplace-v3   --policy.device=cuda   --wandb.enable=true   --policy.repo_id=AriRyo/redblack-pickplace-v3_act-policy --steps=100000

PI05(lerobot)
gray
lerobot-train \
    --dataset.repo_id=AriRyo/gray-pickplace-v3 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_gray-pickplace-v3 \
    --job_name=pi05_gray-pickplace-v3 \
    --policy.repo_id=AriRyo/gray-pickplace-v3_pi05-policy \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --steps=5000 \
    --policy.device=cuda \
    --batch_size=32

redblack
lerobot-train \
    --dataset.repo_id=AriRyo/redblack-pickplace-v3 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_redblack-pickplace-v3 \
    --job_name=pi05_redblack-pickplace-v3 \
    --policy.repo_id=AriRyo/redblack-pickplace-v3_pi05-policy \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --steps=5000 \
    --policy.device=cuda \
    --batch_size=32


Pi0/Pi05 LoRA fine-tuning

scripts/apply_pi0_peft.py \
    --dataset.repo_id=AriRyo/redblack-pickplace-v3 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_redblack_lora \
    --job_name=pi05_redblack_lora \
    --policy.repo_id=AriRyo/redblack-pickplace-v3_pi05-lora \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=32 \
    --lora_push_to_hub=true \
    --lora_repo_id=AriRyo/redblack-pickplace-v3_pi05-lora

LoRA版では自動的にPALIGEMMA/GEMMAにPEFTアダプタを挿入し、学習後に
`<output_dir>/lora_adapters/`へ `paligemma/` と `gemma_expert/` の2フォルダが出力されます。
`--lora_merge_full_policy=true` を付けると LoRA をベースモデルへマージしたチェックポイントも保存されます。



推論のコマンド

rm -rf /home/arifuku/.cache/huggingface/lerobot/AriRyo/eval_gray-pickplace-v2/

lerobot-record     --robot.type=so101_follower     --robot.port=/dev/ttyACM1     --robot.id=my_follower_arm     --robot.cameras="{ above: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}"  --display_data=false     --dataset.repo_id=AriRyo/eval_gray-pickplace-v2     --dataset.num_episodes=3     --dataset.single_task="Pick the gray cube and place it on the circle."     --dataset.episode_time_s=30     --dataset.reset_time_s=10     --dataset.push_to_hub=false --policy.path=AriRyo/gray-pickplace-v2_act-policy --resume=true



データセット上の評価
uv run scripts/eval_test.py
データセットIDとかはハードコードされているので、必要に応じて書き換える



## Data Augmentation実験メモ

### 0. 共通準備
```
export DA_BASE_ARGS="\
    --dataset.repo_id=AriRyo/pickplace-v3_merged \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --wandb.enable=true \
    --steps=60000 \
    --batch_size=32 \
    --num_workers=8 \
    --save_checkpoint=true \
    --eval_freq=0"
```
※ `TrainPipelineConfig` と `ImageTransformsConfig` を確認済み（src/lerobot/configs/train.py, src/lerobot/datasets/transforms.py）。ドット記法は辞書型 (`tfs`) には使えないため、JSON文字列でまとめて渡す点に注意。

### 1. ベースライン (E0)
```
lerobot-train \
    $DA_BASE_ARGS \
    --job_name=act_da_e0_baseline \
    --output_dir=outputs/train/act_da/e0_baseline \
    --policy.repo_id=AriRyo/pickplace-v3_merged_act-da-e0-baseline \
    --dataset.image_transforms.enable=false
```

### 2. 個別拡張 (E1)
共通で `max_num_transforms=1` に設定し、`tfs` には対象となる変換のみを JSON 文字列で渡す。
```
# Brightness only
export TFS_E1_BRIGHTNESS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}}}'
lerobot-train $DA_BASE_ARGS \
    --job_name=act_da_e1_brightness \
    --policy.repo_id=AriRyo/pickplace-v3_merged_act-da-e1-brightness \
    --output_dir=outputs/train/act_da/e1_brightness \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=1 \
    --dataset.image_transforms.tfs="$TFS_E1_BRIGHTNESS"

# Contrast only
export TFS_E1_CONTRAST='{"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}}}'
lerobot-train $DA_BASE_ARGS \
    --job_name=act_da_e1_contrast \
    --policy.repo_id=AriRyo/pickplace-v3_merged_act-da-e1-contrast \
    --output_dir=outputs/train/act_da/e1_contrast \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=1 \
    --dataset.image_transforms.tfs="$TFS_E1_CONTRAST"

# Hue only
export TFS_E1_HUE='{"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}}}'
lerobot-train $DA_BASE_ARGS \
    --job_name=act_da_e1_hue \
    --policy.repo_id=AriRyo/pickplace-v3_merged_act-da-e1-hue \
    --output_dir=outputs/train/act_da/e1_hue \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=1 \
    --dataset.image_transforms.tfs="$TFS_E1_HUE"

# Sharpness only
export TFS_E1_SHARPNESS='{"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}}}'
lerobot-train $DA_BASE_ARGS \
    --job_name=act_da_e1_sharpness \
    --policy.repo_id=AriRyo/pickplace-v3_merged_act-da-e1-sharpness \
    --output_dir=outputs/train/act_da/e1_sharpness \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=1 \
    --dataset.image_transforms.tfs="$TFS_E1_SHARPNESS"

# Random Affine only
export TFS_E1_AFFINE='{"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}}'
lerobot-train $DA_BASE_ARGS \
    --job_name=act_da_e1_affine \
    --policy.repo_id=AriRyo/pickplace-v3_merged_act-da-e1-affine \
    --output_dir=outputs/train/act_da/e1_affine \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=1 \
    --dataset.image_transforms.tfs="$TFS_E1_AFFINE"
```
※ JSON 内のパラメータは `ImageTransformConfig` のデフォルト値を転記。レンジを変える場合は該当値を書き換える。

### 3. 全拡張 (E2)
```
lerobot-train $DA_BASE_ARGS \
    --job_name=act_da_e2_fullstack \
    --policy.repo_id=AriRyo/pickplace-v3_merged_act-da-e2-fullstack \
    --output_dir=outputs/train/act_da/e2_fullstack \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.random_order=true
```

### 4. Top-3組み合わせ (E3-1例)
```
export TFS_E3_TOP3='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":0.5,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},"affine":{"weight":0.5,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}}'
lerobot-train $DA_BASE_ARGS \
    --job_name=act_da_e3_top3 \
    --policy.repo_id=AriRyo/pickplace-v3_merged_act-da-e3-top3 \
    --output_dir=outputs/train/act_da/e3_top3 \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=2 \
    --dataset.image_transforms.tfs="$TFS_E3_TOP3"
```
※ 重みはE1評価の結果で調整。`max_num_transforms=2` で同時適用を管理。

### 5. 画像拡張の可視化
```
lerobot-imgtransform-viz \
    --repo_id=AriRyo/pickplace-v3_merged \
    --episodes='[0]' \
    --image_transforms.enable=true \
    --image_transforms.max_num_transforms=3 
```
`DatasetConfig` をそのままCLIで渡せるので、学習と同じ設定を確認可能。

### 6. 評価手順
学習完了後は `scripts/eval_policy.py` の `policy_id` と `eval_repo` を対象モデルに書き換えて実行。
別リポジトリで評価する場合は `LeRobotDataset(eval_repo)` が読み取れるか事前確認。
