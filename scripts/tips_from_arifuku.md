
テレオペのコマンド


```
lerobot-teleoperate --robot.type=so101_follower   --robot.port=/dev/ttyACM0  --robot.id=my_follower_arm --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_leader_arm
```


データ収集のコマンド

```
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
```

赤黒のデータ収集コマンド

```
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
```


データセットのマージコマンド
```
lerobot-edit-dataset \
    --repo_id AriRyo/pickplace-v3_merged \
    --operation.type merge \
    --operation.repo_ids "['AriRyo/gray-pickplace-v3', 'AriRyo/redblack-pickplace-v3']" \
    --push_to_hub true
```

注意: LeRobot 0.4.1のバグ修正として、src/lerobot/datasets/dataset_tools.pyの259行目で
video_files_size_in_mb=5000を設定しています（デフォルトは500MB）。
これにより、ビデオが複数ファイルに分割されることを防ぎます。
詳細: https://github.com/huggingface/lerobot/issues/2328


学習コマンド
ACT
gray
```
nohup lerobot-train   --dataset.repo_id=AriRyo/gray-pickplace-v3   --policy.type=act   --output_dir=outputs/train/act_gray-pickplace-v3   --job_name=act_gray-pickplace-v3   --policy.device=cuda   --wandb.enable=true   --policy.repo_id=AriRyo/gray-pickplace-v3_act-policy > output.log &
```

redblack
```
lerobot-train   --dataset.repo_id=AriRyo/redblack-pickplace-v3   --policy.type=act   --output_dir=outputs/train/act_redblack-pickplace-v3   --job_name=act_redblack-pickplace-v3   --policy.device=cuda   --wandb.enable=true   --policy.repo_id=AriRyo/redblack-pickplace-v3_act-policy --steps=100000
```

PI05(lerobot)
gray
```
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
```

redblack
```
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
```


Pi0/Pi05 LoRA fine-tuning

```
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
```

LoRA版では自動的にPALIGEMMA/GEMMAにPEFTアダプタを挿入し、学習後に
`<output_dir>/lora_adapters/`へ `paligemma/` と `gemma_expert/` の2フォルダが出力されます。
`--lora_merge_full_policy=true` を付けると LoRA をベースモデルへマージしたチェックポイントも保存されます。



推論のコマンド

```
rm -rf /home/arifuku/.cache/huggingface/lerobot/AriRyo/eval_gray-pickplace-v2/

lerobot-record     --robot.type=so101_follower     --robot.port=/dev/ttyACM1     --robot.id=my_follower_arm     --robot.cameras="{ above: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}"  --display_data=false     --dataset.repo_id=AriRyo/eval_gray-pickplace-v2     --dataset.num_episodes=3     --dataset.single_task="Pick the gray cube and place it on the circle."     --dataset.episode_time_s=30     --dataset.reset_time_s=10     --dataset.push_to_hub=false --policy.path=AriRyo/gray-pickplace-v2_act-policy --resume=true
```



データセット上の評価
uv run scripts/eval_test.py
データセットIDとかはハードコードされているので、必要に応じて書き換える



## Data Augmentation実験メモ

### 0. 共通準備
```
export DA_BASE_ARGS="\
    --dataset.repo_id=AriRyo/pickplace-v4_black \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --dataset.video_backend=pyav \
    --steps=100000 \
    --batch_size=8 \
    --save_checkpoint=true"
```
※ `TrainPipelineConfig` と `ImageTransformsConfig` を確認済み（src/lerobot/configs/train.py, src/lerobot/datasets/transforms.py）。ドット記法は辞書型 (`tfs`) には使えないため、JSON文字列でまとめて渡す点に注意。

```
lerobot-train \
    $DA_BASE_ARGS \
    --job_name=act_da_e0_baseline \
    --output_dir=outputs/train/act_da/e0_baseline \
    --policy.repo_id=AriRyo/pickplace-v4_black_act-da-e0-baseline \
    --dataset.image_transforms.enable=false

lerobot-train $DA_BASE_ARGS \
    --job_name=act_da_e2_fullstack \
    --policy.repo_id=AriRyo/pickplace-v4_black_act-da-e2-fullstack \
    --output_dir=outputs/train/act_da/e2_fullstack \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.random_order=true

# Brightness only
export TFS_E1_BRIGHTNESS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}}}'
lerobot-train $DA_BASE_ARGS \
    --job_name=act_da_e1_brightness \
    --policy.repo_id=AriRyo/pickplace-v4_black_act-da-e1-brightness \
    --output_dir=outputs/train/act_da/e1_brightness \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=1 \
    --dataset.image_transforms.tfs="$TFS_E1_BRIGHTNESS"

# Contrast only
export TFS_E1_CONTRAST='{"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}}}'
lerobot-train $DA_BASE_ARGS \
    --job_name=act_da_e1_contrast \
    --policy.repo_id=AriRyo/pickplace-v4_black_act-da-e1-contrast \
    --output_dir=outputs/train/act_da/e1_contrast \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=1 \
    --dataset.image_transforms.tfs="$TFS_E1_CONTRAST"

# Hue only
export TFS_E1_HUE='{"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}}}'
lerobot-train $DA_BASE_ARGS \
    --job_name=act_da_e1_hue \
    --policy.repo_id=AriRyo/pickplace-v4_black_act-da-e1-hue \
    --output_dir=outputs/train/act_da/e1_hue \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=1 \
    --dataset.image_transforms.tfs="$TFS_E1_HUE"

# Sharpness only
export TFS_E1_SHARPNESS='{"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}}}'
lerobot-train $DA_BASE_ARGS \
    --job_name=act_da_e1_sharpness \
    --policy.repo_id=AriRyo/pickplace-v4_black_act-da-e1-sharpness \
    --output_dir=outputs/train/act_da/e1_sharpness \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=1 \
    --dataset.image_transforms.tfs="$TFS_E1_SHARPNESS"

# Random Affine only
export TFS_E1_AFFINE='{"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}}'
lerobot-train $DA_BASE_ARGS \
    --job_name=act_da_e1_affine \
    --policy.repo_id=AriRyo/pickplace-v4_black_act-da-e1-affine \
    --output_dir=outputs/train/act_da/e1_affine \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=1 \
    --dataset.image_transforms.tfs="$TFS_E1_AFFINE"


```

### 4. Top-3組み合わせ (E3-1例)
```
export TFS_E3_TOP3='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":0.5,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},"affine":{"weight":0.5,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}}'
lerobot-train $DA_BASE_ARGS \
    --job_name=act_da_e3_top3 \
    --policy.repo_id=AriRyo/pickplace-v4_black_act-da-e3-top3 \
    --output_dir=outputs/train/act_da/e3_top3 \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=2 \
    --dataset.image_transforms.tfs="$TFS_E3_TOP3"
```
※ 重みはE1評価の結果で調整。`max_num_transforms=2` で同時適用を管理。

```
# LoRAファインチューニングのサンプルコマンド
python scripts/train_pi05_lora.py \
    --dataset.repo_id=AriRyo/pickplace-v4_black \
    --policy.type=pi05 \
    --output_dir=./outputs/train/peft/pi05_lora_training \
    --job_name=pi05_lora_training \
    --policy.repo_id=AriRyo/pickplace-v4_black_pi05_lora \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=false \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --policy.dtype=bfloat16 \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=16

```

このスクリプトでは、
make_policy で pi05 ベースモデルをロード
すぐ後に apply_pi05_lora(policy) で LoRA を挿入し、lora_ だけ requires_grad=True
その状態で make_optimizer_and_scheduler を呼ぶので、optimizer は LoRA パラメータだけを更新
学習ループ・eval・チェックポイント保存は lerobot_train.py と同じ


nohupコマンド
nohup コマンド > output.log 2>&1 &


    --wandb.disable_artifact=true
を付けると、WandBのアーティファクト保存を無効化できるっぽい


python scripts/train_pi05_lora.py \
    --dataset.repo_id=AriRyo/pickplace-v4_black \
    --policy.type=pi05 \
    --output_dir=./outputs/train/pi05-lora_pickplace-v4_black \
    --job_name=pi05-lora_pickplace-v4_black \
    --policy.repo_id=AriRyo/pickplace-v4_black_pi05_lora \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=false \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --policy.dtype=bfloat16 \
    --steps=20000 \
    --policy.device=cuda \
    --batch_size=7

#そのままのpy05学習コマンド
lerobot-train \
    --dataset.repo_id=AriRyo/pickplace-v4_black \
    --policy.type=pi05 \
    --output_dir=./outputs/train/pi05_pickplace-v4_black \
    --job_name=pi05_pickplace-v4_black \
    --policy.repo_id=AriRyo/pi05_pickplace-v4_black \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --policy.dtype=bfloat16 \
    --steps=5000 \
    --policy.device=cuda \
    --batch_size=32



# データ拡張のやつ
# 共通パラメータ（ACT）


XVLA
lerobot-train   --dataset.repo_id=AriRyo/pickplace-v4_black   --output_dir=./outputs/train/pickplace-v4_black_xvla_tf-sp   --job_name=pickplace-v4_black_xvla_tf-sp   --policy.path="lerobot/xvla-base"   --policy.repo_id="AriRyo/pickplace-v4_black_xvla_tf-sp"   --steps=6000   --policy.device=cuda   --policy.freeze_vision_encoder=true   --policy.freeze_language_encoder=true   --policy.train_policy_transformer=true   --policy.train_soft_prompts=true   --policy.action_mode=auto --rename_map='{"observation.images.above": "observation.images.image", "observation.images.side":  "observation.images.image2"}'  --dataset.video_backend=pyav --wandb.enable=true --wandb.disable_artifact=true

lerobot-train   --dataset.repo_id=AriRyo/pickplace-v4_black   --output_dir=./outputs/train/pickplace-v4_black_xvla_tf-sp_10k   --job_name=pickplace-v4_black_xvla_tf-sp   --policy.path="lerobot/xvla-base"   --policy.repo_id="AriRyo/pickplace-v4_black_xvla_tf-sp_10k"   --steps=10000   --policy.device=cuda   --policy.freeze_vision_encoder=true   --policy.freeze_language_encoder=true   --policy.train_policy_transformer=true   --policy.train_soft_prompts=true   --policy.action_mode=auto --rename_map='{"observation.images.above": "observation.images.image", "observation.images.side":  "observation.images.image2"}'  --dataset.video_backend=pyav --wandb.enable=true --wandb.disable_artifact=true

lerobot-train   --dataset.repo_id=AriRyo/pickplace-v4_black   --output_dir=./outputs/train/pickplace-v4_black_xvla_sp_10k   --job_name=pickplace-v4_black_xvla_sp   --policy.path="lerobot/xvla-base"   --policy.repo_id="AriRyo/pickplace-v4_black_xvla_sp_10k"   --steps=10000   --policy.device=cuda   --policy.freeze_vision_encoder=true   --policy.freeze_language_encoder=true   --policy.train_policy_transformer=false   --policy.train_soft_prompts=true   --policy.action_mode=auto --rename_map='{"observation.images.above": "observation.images.image", "observation.images.side":  "observation.images.image2"}'  --dataset.video_backend=pyav --wandb.enable=true --wandb.disable_artifact=true

lerobot-train   --dataset.repo_id=AriRyo/pickplace-v4_black   --output_dir=./outputs/train/pickplace-v4_black_xvla_tf_10k   --job_name=pickplace-v4_black_xvla_tf   --policy.path="lerobot/xvla-base"   --policy.repo_id="AriRyo/pickplace-v4_black_xvla_tf_10k"   --steps=10000   --policy.device=cuda   --policy.freeze_vision_encoder=true   --policy.freeze_language_encoder=true   --policy.train_policy_transformer=true   --policy.train_soft_prompts=false   --policy.action_mode=auto --rename_map='{"observation.images.above": "observation.images.image", "observation.images.side":  "observation.images.image2"}'  --dataset.video_backend=pyav --wandb.enable=true --wandb.disable_artifact=true




# 共通パラメータ（ACT）
export CLIP_ACT_ARGS="\
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --dataset.video_backend=pyav \
  --steps=100000 \
  --save_checkpoint=true" 

# mask 20%
lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_mask_20   --job_name=pickplace-v4_black_mask_20_act   --output_dir=outputs/train/act_da_grid/pickplace-v4_black_mask_20   --policy.repo_id=AriRyo/pickplace-v4_black_mask_20_act-policy
# mask 50%
lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_mask_50   --job_name=pickplace-v4_black_mask_50_act   --output_dir=outputs/train/act_da_grid/pickplace-v4_black_mask_50   --policy.repo_id=AriRyo/pickplace-v4_black_mask_50_act-policy
# mask 80%
lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_mask_80   --job_name=pickplace-v4_black_mask_80_act   --output_dir=outputs/train/act_da_grid/pickplace-v4_black_mask_80   --policy.repo_id=AriRyo/pickplace-v4_black_mask_80_act-policy
# mask 100%
lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_mask_100   --job_name=pickplace-v4_black_mask_100_act   --output_dir=outputs/train/act_da_grid/pickplace-v4_black_mask_100   --policy.repo_id=AriRyo/pickplace-v4_black_mask_100_act-policy
# mosaic 20%
lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_mosaic_20   --job_name=pickplace-v4_black_mosaic_20_act   --output_dir=outputs/train/act_da_grid/pickplace-v4_black_mosaic_20   --policy.repo_id=AriRyo/pickplace-v4_black_mosaic_20_act-policy
# mosaic 50%
lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_mosaic_50   --job_name=pickplace-v4_black_mosaic_50_act   --output_dir=outputs/train/act_da_grid/pickplace-v4_black_mosaic_50   --policy.repo_id=AriRyo/pickplace-v4_black_mosaic_50_act-policy
# mosaic 80%
lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_mosaic_80   --job_name=pickplace-v4_black_mosaic_80_act   --output_dir=outputs/train/act_da_grid/pickplace-v4_black_mosaic_80   --policy.repo_id=AriRyo/pickplace-v4_black_mosaic_80_act-policy
# mosaic 100%
lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_mosaic_100   --job_name=pickplace-v4_black_mosaic_100_act   --output_dir=outputs/train/act_da_grid/pickplace-v4_black_mosaic_100   --policy.repo_id=AriRyo/pickplace-v4_black_mosaic_100_act-policy
# swap 20%
lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_grid_swap_20   --job_name=pickplace-v4_black_grid_swap_20_act   --output_dir=outputs/train/act_da_grid/pickplace-v4_black_grid_swap_20   --policy.repo_id=AriRyo/pickplace-v4_black_grid_swap_20_act-policy
# swap 50%
lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_grid_swap_50   --job_name=pickplace-v4_black_grid_swap_50_act   --output_dir=outputs/train/act_da_grid/pickplace-v4_black_grid_swap_50   --policy.repo_id=AriRyo/pickplace-v4_black_grid_swap_50_act-policy
# swap 80%
lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_grid_swap_80   --job_name=pickplace-v4_black_grid_swap_80_act   --output_dir=outputs/train/act_da_grid/pickplace-v4_black_grid_swap_80   --policy.repo_id=AriRyo/pickplace-v4_black_grid_swap_80_act-policy
# swap 100%
lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_grid_swap_100   --job_name=pickplace-v4_black_grid_swap_100_act   --output_dir=outputs/train/act_da_grid/pickplace-v4_black_grid_swap_100   --policy.repo_id=AriRyo/pickplace-v4_black_grid_swap_100_act-policy




lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_da_colored_grid_fill_5   --job_name=pickplace-v4_black_grid_fill_5_act   --output_dir=outputs/train/act_da_colored/pickplace-v4_black_grid_fill_5   --policy.repo_id=AriRyo/pickplace-v4_black_colored_grid_fill_5_act

lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_da_colored_grid_overlay_5   --job_name=pickplace-v4_black_grid_overlay_5_act   --output_dir=outputs/train/act_da_colored/pickplace-v4_black_grid_overlay_5   --policy.repo_id=AriRyo/pickplace-v4_black_colored_grid_overlay_5_act


lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_da_colored_graph_fill_5   --job_name=pickplace-v4_black_graph_fill_5_act   --output_dir=outputs/train/act_da_colored/pickplace-v4_black_graph_fill_5   --policy.repo_id=AriRyo/pickplace-v4_black_colored_graph_fill_5_act

lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_da_colored_graph_overlay_5   --job_name=pickplace-v4_black_graph_overlay_5_act   --output_dir=outputs/train/act_da_colored/pickplace-v4_black_graph_overlay_5   --policy.repo_id=AriRyo/pickplace-v4_black_colored_graph_overlay_5_act


lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_da_colored_SLIC_fill_5   --job_name=pickplace-v4_black_slic_fill_5_act   --output_dir=outputs/train/act_da_colored/pickplace-v4_black_slic_fill_5   --policy.repo_id=AriRyo/pickplace-v4_black_colored_slic_fill_5_act

lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_da_colored_SLIC_overlay_5   --job_name=pickplace-v4_black_slic_overlay_5_act   --output_dir=outputs/train/act_da_colored/pickplace-v4_black_slic_overlay_5   --policy.repo_id=AriRyo/pickplace-v4_black_colored_slic_overlay_5_act


lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_da_colored_ms_fill_5   --job_name=pickplace-v4_black_meanshift_fill_5_act   --output_dir=outputs/train/act_da_colored/pickplace-v4_black_meanshift_fill_5   --policy.repo_id=AriRyo/pickplace-v4_black_colored_meanshift_fill_5_act

lerobot-train   $CLIP_ACT_ARGS   --dataset.repo_id=AriRyo/pickplace-v4_black_da_colored_ms_overlay_5   --job_name=pickplace-v4_black_meanshift_overlay_5_act   --output_dir=outputs/train/act_da_colored/pickplace-v4_black_meanshift_overlay_5   --policy.repo_id=AriRyo/pickplace-v4_black_colored_meanshift_overlay_5_act





