"""
Weights & Biases callback 
"""

import os
import datetime
from typing import Any, Dict

import torch
import wandb

from beam.training.callbacks import Callback
from beam.utils.wandb_utils import (
    init_wandb, log_metrics, log_sample_images, finish_run
)


class WeightsAndBiasesCallback(Callback):
    """
    Callback for logging to Weights & Biases.

    Args:
        project_name: W&B project name
        model_name: Name of the model for W&B
        log_freq: Frequency (in batches) for logging batch metrics
        log_samples_freq: Frequency (in epochs) for logging sample images
        log_model_freq: Frequency (in epochs) for logging model weights
        use_wandb: Whether to use W&B (set to False to disable)
    """

    def __init__(
        self,
        project_name: str = "beam-tess",
        model_name: str = None,
        log_freq: int = 10,
        log_samples_freq: int = 5,
        log_model_freq: int = 20,
        use_wandb: bool = True,
        mode: str = "online",
        MEAN: float = None,
        STD: float = None,
    ):
        super().__init__()
        self.project_name = project_name
        self.model_name = model_name
        self.log_freq = log_freq
        self.log_samples_freq = log_samples_freq
        self.log_model_freq = log_model_freq
        self.use_wandb = use_wandb
        self.mode = mode if use_wandb else "disabled"
        self.batch_count = 0
        self.current_step = 0  # Track the current step for consistent logging
        self.MEAN = MEAN
        self.STD = STD

    def on_train_begin(self, trainer: Any) -> None:
        """Initialize W&B at the start of training."""
        if not self.use_wandb:
            return

        config = trainer.config.copy()
        if self.model_name is None:
            time_str = datetime.datetime.fromtimestamp(
                trainer.start_training_time).strftime("%Y%m%d_%H%M%S")
            model_name = config.get('wandb_model_name', 'beam-tess-model')
            self.model_name = f"{model_name}_{time_str}"

        config["run_name"] = self.model_name
        init_wandb(config, self.project_name, self.mode)

        print(f"Initialized W&B logging with run name: {self.model_name}")

        self.steps_per_epoch = len(trainer.train_loader)
        if trainer.current_epoch > 0:
            self.current_step = trainer.current_epoch * self.steps_per_epoch

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """Reset batch counter at the beginning of each epoch."""
        self.batch_count = 0

    #unused
    def _log_model_fixed(self, trainer: Any, name: str) -> None:
        # model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        # state = {
        #     'model_state_dict': model.state_dict(),
        #     'ema_state_dict': trainer.ema.state_dict(),
        # }
        # # Save inside the run dir: wandb owns this directory, the file persists
        # # for the whole run, and policy="now" uploads immediately.
        # path = os.path.join(wandb.run.dir, f"{name}.pt")
        # torch.save(state, path)
        # wandb.save(path, base_path=wandb.run.dir, policy="now")
        return

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict) -> None:
        """Full reimplementation of V1's on_epoch_end (not super()) so the
        broken log_model call can be replaced; metrics/sample logging follow
        V1's behavior plus the t-bin diagnostics."""
        if not self.use_wandb:
            return

        self.current_step = max(self.current_step, epoch * self.steps_per_epoch + self.batch_count)

        # Epoch metrics
        try:
            metrics = {
                "epoch": epoch,
                "train_loss": trainer.loss_history_train[-1],
                "valid_loss": trainer.loss_history_valid[-1],
                "time_hours": trainer.time_history[-1] if trainer.time_history else 0,
            }
            if hasattr(trainer, 'optimizer'):
                metrics["learning_rate"] = trainer.optimizer.param_groups[0]['lr']

            metrics.update({k: v for k, v in logs.items()
                            if k.startswith(("valid_loss_t_bin_", "valid_flux_err_t_bin_",
                                             "valid_loss_sigma_bin_", "valid_flux_err_sigma_bin_"))})
            if "best_valid_loss" in logs:
                metrics["best_valid_loss"] = logs["best_valid_loss"]
            log_metrics(metrics, step=self.current_step)
        except Exception as e:
            print(f"Warning: Failed to log epoch metrics to W&B: {e}")

        # Sample images
        if epoch % self.log_samples_freq == 0 or epoch == trainer.config['training_n_epoch'] - 1:
            try:
                model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
                image_shape = (trainer.config['data_image_shape']
                               if trainer.config['data_patch_size'] is None
                               else trainer.config.get('data_patch_size'))
                for loader, tag in ((trainer.train_loader, "Training Set"),
                                    (trainer.valid_loader, "Validation Set")):
                    log_sample_images(
                        model=model,
                        dataloader=loader,
                        device=trainer.device,
                        n_sample=1,
                        n_datapoint=1,
                        image_shape=image_shape,
                        step=self.current_step,
                        name=tag,
                        ema=trainer.ema,
                        MEAN=self.MEAN,
                        STD=self.STD,
                    )
            except Exception as e:
                print(f"Warning: Failed to log sample images to W&B: {e}")

        # Model weights 
        if epoch % self.log_model_freq == 0 or epoch == trainer.config['training_n_epoch'] - 1:
            try:
                self._log_model_fixed(trainer, f"model_epoch_{epoch}")
            except Exception as e:
                print(f"Warning: Failed to log model weights to W&B: {e}")

    def on_batch_end(self, trainer: Any, batch: int, logs: Dict) -> None:
        """Log batch metrics periodically."""
        # Increment batch counter even if not logging
        self.batch_count += 1

        self.current_step = max(
            self.current_step,
            trainer.current_epoch * self.steps_per_epoch + self.batch_count)

        if not self.use_wandb or batch % self.log_freq != 0:
            return

        # Only log every log_freq batches and avoid overwhelming the server
        try:
            metrics = {
                "batch": self.batch_count,
                "batch_loss": logs.get("batch_loss", 0),
            }
            log_metrics(metrics, step=self.current_step)
        except Exception as e:
            print(f"Warning: Failed to log batch metrics to W&B: {e}")
            # Continue training even if logging fails
            pass

    def on_train_end(self, trainer: Any) -> None:
        """Finish the W&B run at the end of training, with the fixed saver."""
        if not self.use_wandb:
            return
        try:
            self._log_model_fixed(trainer, "final_model")
        except Exception as e:
            print(f"Warning: Failed to log final model to W&B: {e}")
        finish_run()
