#!/usr/bin/env python
"""Offline LoRA fine-tuning entry point for Pi0/Pi05 policies.

This script mirrors ``lerobot-train`` so that you can keep using commands such as::

    python scripts/train_pi0_peft.py \
        --dataset.repo_id=AriRyo/pickplace-v3_merged \
        --policy.type=pi05 \
        --output_dir=./outputs/train/peft/pi05_lora \
        --job_name=pi05_lora \
        --policy.repo_id=AriRyo/pickplace-v3_merged_pi05-lora \
        --policy.pretrained_path=lerobot/pi05_base \
        --policy.compile_model=true \
        --policy.gradient_checkpointing=true \
        --wandb.enable=true \
        --policy.dtype=bfloat16 \
        --steps=3 \
        --policy.device=cuda \
        --batch_size=32

Compared to ``lerobot-train`` the policy automatically receives OpenPI-style LoRA adapters,
non-LoRA parameters are frozen, and the trained adapters are exported (and optionally pushed
to the Hub) once training finishes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from huggingface_hub import HfApi
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored
from torch.utils.data import DataLoader

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, init_logging

from scripts.pi0_lora_utils import (
    LoRAHandles,
    attach_lora_adapters,
    export_lora_adapters,
    freeze_non_lora,
)


@dataclass
class Pi0LoraTrainConfig(TrainPipelineConfig):
    """TrainPipelineConfig with extra knobs for LoRA fine-tuning."""

    lora_zero_init: bool = True
    lora_init_path: Path | None = None
    lora_rank_override: int | None = None
    lora_alpha_override: float | None = None
    lora_export_dir: Path | None = None
    lora_merge_full_policy: bool = False
    merged_policy_dir: Path | None = None
    lora_push_to_hub: bool = False
    lora_repo_id: str | None = None
    lora_private: bool | None = None
    lora_token: str | None = None
    lora_branch: str | None = None

    resume_pretrained_dir: Path | None = field(init=False, default=None)
    policy_pretrained_snapshot: Path | str | None = field(init=False, default=None)

    def validate(self) -> None:
        super().validate()

        if self.policy.type not in {"pi0", "pi05"}:
            raise ValueError("This entry point only supports Pi0 / Pi05 policies.")

        self.policy_pretrained_snapshot = self.policy.pretrained_path

        if self.resume and self.policy.pretrained_path:
            self.resume_pretrained_dir = Path(self.policy.pretrained_path)
            # Instantiate the policy fresh and load the checkpoint manually once LoRA is attached
            self.policy.pretrained_path = None

        if self.output_dir is not None:
            if self.lora_export_dir is None:
                self.lora_export_dir = self.output_dir / "lora_adapters"
            if self.lora_merge_full_policy and self.merged_policy_dir is None:
                self.merged_policy_dir = self.output_dir / "merged_policy"

        if self.lora_push_to_hub and not self.lora_repo_id:
            raise ValueError("--lora_repo_id must be provided when --lora_push_to_hub is true")


def _load_policy_state(policy: PreTrainedPolicy, pretrained_dir: Path, map_location: str | None) -> None:
    model_file = pretrained_dir / SAFETENSORS_SINGLE_FILE
    if not model_file.exists():
        raise FileNotFoundError(model_file)
    policy.__class__._load_as_safetensor(policy, str(model_file), map_location or "cpu", strict=True)


def _instantiate_policy_with_lora(cfg: Pi0LoraTrainConfig, dataset_meta) -> tuple[PreTrainedPolicy, LoRAHandles]:
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset_meta,
        rename_map=cfg.rename_map,
    )

    handles = attach_lora_adapters(
        policy,
        rank_override=cfg.lora_rank_override,
        alpha_override=cfg.lora_alpha_override,
        zero_init=cfg.lora_zero_init,
        init_adapter_dir=cfg.lora_init_path,
    )
    freeze_non_lora(policy)
    setattr(policy, "_lora_handles", handles)

    if cfg.resume_pretrained_dir is not None:
        _load_policy_state(policy, cfg.resume_pretrained_dir, cfg.policy.device)

    return policy, handles


def _push_lora_folder_to_hub(folder: Path, cfg: Pi0LoraTrainConfig) -> None:
    api = HfApi(token=cfg.lora_token)
    repo_id = api.create_repo(repo_id=cfg.lora_repo_id, private=cfg.lora_private, exist_ok=True).repo_id
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        commit_message="Upload Pi0/Pi05 LoRA adapters",
        revision=cfg.lora_branch,
        repo_type="model",
    )


def _export_lora_artifacts(
    accelerator: Accelerator,
    cfg: Pi0LoraTrainConfig,
    policy: PreTrainedPolicy,
) -> None:
    if cfg.lora_export_dir is None:
        return

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    base_policy = accelerator.unwrap_model(policy)
    handles: LoRAHandles | None = getattr(base_policy, "_lora_handles", None)
    if handles is None:
        logging.warning("LoRA handles missing on policy, skipping adapter export")
        return

    export_dir = Path(cfg.lora_export_dir)
    export_lora_adapters(base_policy, handles, export_dir)
    logging.info("Saved LoRA adapters to %s", export_dir)

    if cfg.lora_merge_full_policy:
        handles.paligemma.merge_and_unload()
        handles.gemma_expert.merge_and_unload()
        merged_dir = Path(cfg.merged_policy_dir or (export_dir.parent / "merged_policy"))
        merged_dir.mkdir(parents=True, exist_ok=True)
        base_policy.save_pretrained(merged_dir)
        logging.info("Saved merged policy snapshot to %s", merged_dir)

    if cfg.lora_push_to_hub:
        _push_lora_folder_to_hub(export_dir, cfg)
        logging.info("Uploaded LoRA adapters to %s", cfg.lora_repo_id)


def _build_processors(cfg: Pi0LoraTrainConfig, dataset, policy: PreTrainedPolicy):
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    pretrained_path_for_processors = cfg.resume_pretrained_dir or cfg.policy_pretrained_snapshot

    if pretrained_path_for_processors is not None:
        processor_kwargs.setdefault("preprocessor_overrides", {})
        processor_kwargs["preprocessor_overrides"].update(
            {
                "device_processor": {"device": cfg.policy.device},
                "normalizer_processor": {
                    "stats": dataset.meta.stats,
                    "features": {**policy.config.input_features, **policy.config.output_features},
                    "norm_map": policy.config.normalization_mapping,
                },
                "rename_observations_processor": {"rename_map": cfg.rename_map},
            }
        )

        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            }
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=pretrained_path_for_processors,
        **processor_kwargs,
        **postprocessor_kwargs,
    )
    return preprocessor, postprocessor


@parser.wrap()
def train(cfg: Pi0LoraTrainConfig, accelerator: Accelerator | None = None):  # noqa: PLR0912, PLR0915
    cfg.validate()

    if accelerator is None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process

    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    if not is_main_process:
        dataset = make_dataset(cfg)

    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        if is_main_process:
            logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy with LoRA adapters")
    policy, _ = _instantiate_policy_with_lora(cfg, dataset.meta)

    accelerator.wait_for_everyone()

    preprocessor, postprocessor = _build_processors(cfg, dataset, policy)

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info("Start offline training with LoRA adapters")

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,
                        policy=accelerator.unwrap_model(policy),
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                aggregated = eval_info["overall"]

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    _export_lora_artifacts(accelerator, cfg, policy)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    train()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
