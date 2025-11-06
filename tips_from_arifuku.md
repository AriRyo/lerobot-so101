





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



推論のコマンド

rm -rf /home/arifuku/.cache/huggingface/lerobot/AriRyo/eval_gray-pickplace-v2/

lerobot-record     --robot.type=so101_follower     --robot.port=/dev/ttyACM1     --robot.id=my_follower_arm     --robot.cameras="{ above: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}"  --display_data=false     --dataset.repo_id=AriRyo/eval_gray-pickplace-v2     --dataset.num_episodes=3     --dataset.single_task="Pick the gray cube and place it on the circle."     --dataset.episode_time_s=30     --dataset.reset_time_s=10     --dataset.push_to_hub=false --policy.path=AriRyo/gray-pickplace-v2_act-policy --resume=true



データセット上の評価
uv run scripts/eval_test.py
データセットIDとかはハードコードされているので、必要に応じて書き換える


