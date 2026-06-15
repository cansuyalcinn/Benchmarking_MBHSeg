
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from torch import autocast, nn
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.collate_outputs import collate_outputs
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
import torch
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
import warnings
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.crossval_split import generate_crossval_split
from torch.optim import Adam
from torch import distributed as dist
import copy
from scipy.ndimage import binary_erosion
from collections import OrderedDict
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from time import time, sleep
import torch.nn.functional as F
import shutil
import multiprocessing
import os
import sys 
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from torch.utils.data import Sampler
import itertools
from nnunetv2.training.logging.nnunet_logger import nnUNetLoggerCBS
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from torch._dynamo import OptimizedModule

# UAMT (Uncertainty-Aware Mean Teacher) model.
# Contains one student network (trained with backprop) and one teacher network (EMA of student).
# The teacher generates uncertainty-weighted pseudo labels for the unlabeled data.
# We use the student network for inference and validation (by default nnunet).


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss.
    
    Used as the consistency criterion between student and teacher predictions.
    Returns per-element MSE (no reduction), so we can apply uncertainty masking.
    
    Args:
        input_logits: Student network logits. Shape: (B, C, ...)  
        target_logits: Teacher network logits. Shape: (B, C, ...)
    
    Returns:
        Per-element MSE loss. Shape: (B, C, ...)
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss


class CombinedSemiSupervisedDataloader:
    def __init__(self, dataloader_labeled, dataloader_unlabeled):
        self.dataloader_labeled = dataloader_labeled
        self.dataloader_unlabeled = dataloader_unlabeled
        
        self.labeled_iter = iter(dataloader_labeled)
        self.unlabeled_iter = iter(dataloader_unlabeled)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch_l = next(self.labeled_iter)
        except StopIteration:
            self.labeled_iter = iter(self.dataloader_labeled)
            batch_l = next(self.labeled_iter)

        try:
            batch_u = next(self.unlabeled_iter)
        except StopIteration:
            self.unlabeled_iter = iter(self.dataloader_unlabeled)
            batch_u = next(self.unlabeled_iter)

        # Combine batches: [labeled, unlabeled]
        combined = {
            "data": torch.cat([batch_l["data"], batch_u["data"]], dim=0),
            "target": torch.cat([batch_l["target"], batch_u["target"]], dim=0)
        }
        return combined


class nnUNetTrainerUAMT(nnUNetTrainerNoDeepSupervision):
    """
    Trainer for semi-supervised learning with Uncertainty-Aware Mean Teacher (UAMT).
    
    Key differences from CPS:
    - Instead of two independently trained networks, uses a student-teacher paradigm.
    - The teacher network is an exponential moving average (EMA) of the student.
    - Consistency loss is weighted by uncertainty estimated from multiple stochastic 
      forward passes through the teacher.
    - Only the student network is trained via backpropagation; teacher is updated via EMA.
    
    Paper: "Uncertainty-aware Self-ensembling Model for Semi-supervised 3D Left Atrium Segmentation"
    Yu et al., MICCAI 2019.
    """
    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            device: torch.device = torch.device('cuda'),
            percentage_labeled_data: float = 1.0, 
            seed_name: str = '0', 
            seed_balance = '42'
        ):
        super().__init__(plans, configuration, fold, dataset_json, device, percentage_labeled_data, seed_name, seed_balance)

        # Toggle mixed-precision (autocast) for UAMT training.
        self.use_autocast = True

        # NNunet base settings
        self.percentage_labeled_data = percentage_labeled_data
        self.seed_name = seed_name
        self.seed_balance = seed_balance
        print(f"Percentage of labeled data: {self.percentage_labeled_data}")
        
        ### Hyperparameters (same as CPS baseline)
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 1000
        self.current_epoch = 0
        self.enable_deep_supervision = False

        ### UAMT-specific hyperparameters
        self.ema_decay = 0.99  # EMA decay rate for teacher update
        self.T = 8  # Number of stochastic forward passes for uncertainty estimation
        self.noise_scale = 0.1  # Scale of Gaussian noise added to inputs
        self.noise_clamp = 0.2  # Clamp range for noise

        # Global step counter for EMA warm-up
        self.global_step = 0

        self.logger = nnUNetLoggerCBS()

    def get_current_consistency_weight(self, start_epoch=0, end_epoch=1000, max_weight=0.1):
        """
        Sigmoid ramp-up of consistency weight.
        Gradually increases from 0 to max_weight between start_epoch and end_epoch.
        """
        def sigmoid_rampup(current, rampup_length):
            if rampup_length == 0:
                return 1.0
            else:
                current = np.clip(current, 0.0, rampup_length)
                phase = 1.0 - current / rampup_length
                return float(np.exp(-5.0 * phase * phase))

        rampup_length = end_epoch - start_epoch
        current = self.current_epoch - start_epoch

        if self.current_epoch < start_epoch:
            return 0.0
        elif self.current_epoch >= end_epoch:
            return max_weight
        else:
            return max_weight * sigmoid_rampup(current, rampup_length)

    def get_uncertainty_threshold(self, num_classes: int = None):
        def sigmoid_rampup(current, rampup_length):
            if rampup_length == 0:
                return 1.0
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

        if num_classes is None:
            num_classes = self.label_manager.num_segmentation_heads
        num_classes = max(2, int(num_classes))

        rampup_value = sigmoid_rampup(self.current_epoch, self.num_epochs)
        threshold = (0.75 + 0.25 * rampup_value) * np.log(num_classes)
        return threshold

    @staticmethod
    def update_ema_variables(model, ema_model, alpha, global_step):
        """
        Update teacher (EMA) model parameters.
        
        Uses the true average until the exponential average is more correct.
        alpha = min(1 - 1/(step+1), alpha)
        ema_param = alpha * ema_param + (1 - alpha) * param
        """
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def configure_optimizers(self):
        """Only one optimizer for the student network. Teacher is updated via EMA."""
        optimizer = torch.optim.SGD(self.network.parameters(), 
                                    self.initial_lr, 
                                    weight_decay=self.weight_decay,
                                    momentum=0.99, 
                                    nesterov=True)
        
        lr_scheduler = PolyLRScheduler(optimizer, 
                                       self.initial_lr, 
                                       self.num_epochs)
        return optimizer, lr_scheduler

    def initialize(self):
        if not self.was_initialized:
            self._set_batch_size_and_oversample()
            self.num_input_channels = determine_num_input_channels(self.plans_manager, 
                                                                   self.configuration_manager,
                                                                   self.dataset_json)
            # Student network (trained via backprop)
            self.network = self.build_network_architecture(
                self.plans_manager,
                self.dataset_json,
                self.configuration_manager,
                self.num_input_channels,
                enable_deep_supervision=False
            ).to(self.device)

            # Teacher network (EMA of student) - same architecture, different weights
            self.ema_network = self.build_network_architecture(
                self.plans_manager,
                self.dataset_json,
                self.configuration_manager,
                self.num_input_channels,
                enable_deep_supervision=False
            ).to(self.device)

            # Detach teacher parameters - no gradient computation for EMA model
            for param in self.ema_network.parameters():
                param.detach_()
            
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)
                self.ema_network = torch.compile(self.ema_network)

            # Only one optimizer (student only)
            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])
                self.ema_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.ema_network)
                self.ema_network = DDP(self.ema_network, device_ids=[self.local_rank])

            self.loss = self._build_loss()

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                                "That should not happen.")

    def on_train_epoch_start(self):
        self.network.train()
        self.ema_network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def do_split(self):
        """
        Custom split for semi-supervised training.
        Returns labeled keys, unlabeled keys, and validation keys.
        For 100% labeled, returns (tr_keys, val_keys) matching base nnUNet behavior.
        """
        from nnunetv2.paths import nnUNet_preprocessed
        from os.path import join, dirname
        import json
        
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        if self.fold == "all":
            case_identifiers = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            dataset = self.dataset_class(self.preprocessed_dataset_folder,
                                         identifiers=None,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.identifiers)))
                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)
            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

            self.print_to_log_file("Desired fold for training: %d" % self.fold)

            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file(
                    "This split has %d training and %d validation cases."
                    % (len(tr_keys), len(val_keys))
                )

                if self.percentage_labeled_data == 1.0:
                    self.print_to_log_file(
                        "Using 100% labeled data: falling back to default nnU-Net split."
                    )
                    tr_keys_limit = tr_keys
                    return tr_keys_limit, val_keys
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.identifiers))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')

            parent_dir = dirname(nnUNet_preprocessed)
            split_file = join(
                parent_dir,
                "all_splits_trainset",
                f"balanced_labeled_splits_fold_{self.seed_name}.json"
            )

            with open(split_file, "r") as f:
                balanced_splits = json.load(f)
                print(f"Using balanced splits from {split_file}")

            if self.percentage_labeled_data == 0.1:
                percentage_key = "10_percent"
            elif self.percentage_labeled_data == 0.2:
                percentage_key = "20_percent"
            elif self.percentage_labeled_data == 0.3:
                percentage_key = "30_percent"
            else:
                raise ValueError(
                    f"Unsupported labeled percentage (must be 0.1, 0.2, 0.3, or 1.0): "
                    f"{self.percentage_labeled_data}"
                )

            dataset_name_in_split = self.plans_manager.dataset_name
            labeled_cases = balanced_splits[dataset_name_in_split][f"fold_{self.seed_name}"][percentage_key][f"seed_{self.seed_balance}"]['labeled']
            unlabeled_cases = balanced_splits[dataset_name_in_split][f"fold_{self.seed_name}"][percentage_key][f"seed_{self.seed_balance}"]['unlabeled']

            self.print_to_log_file(f"Using {len(labeled_cases)} labeled cases "f"({percentage_key}, balanced) from {split_file}")

            tr_keys_limit = labeled_cases
            tr_unlabeled_keys = unlabeled_cases

        self.print_to_log_file(f"Using {len(labeled_cases)} labeled training cases out of {len(tr_keys)} total training cases. {len(unlabeled_cases)} unlabeled training cases. {len(val_keys)} validation cases")

        return tr_keys_limit, tr_unlabeled_keys, val_keys
    
    
    def get_tr_and_val_datasets(self):
        tr_labeled_keys, tr_unlabeled_keys, val_keys = self.do_split()

        dataset_tr_labeled = self.dataset_class(self.preprocessed_dataset_folder, tr_labeled_keys,
                                                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        dataset_tr_unlabeled = self.dataset_class(self.preprocessed_dataset_folder, tr_unlabeled_keys,
                                                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        return dataset_tr_labeled, dataset_tr_unlabeled, dataset_val


    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr_labeled_split, dataset_tr_unlabeled_split, dataset_val_split = self.get_tr_and_val_datasets()

        dl_tr_labeled = nnUNetDataLoader(dataset_tr_labeled_split, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)

        dl_tr_unlabeled = nnUNetDataLoader(dataset_tr_unlabeled_split, self.batch_size,
                                initial_patch_size,
                                self.configuration_manager.patch_size,
                                self.label_manager,
                                oversample_foreground_percent=self.oversample_foreground_percent,
                                sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                probabilistic_oversampling=self.probabilistic_oversampling)
                
        dl_val = nnUNetDataLoader(dataset_val_split, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)
        
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train_l = SingleThreadedAugmenter(dl_tr_labeled, None)
            mt_gen_train_u = SingleThreadedAugmenter(dl_tr_unlabeled, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train_l = NonDetMultiThreadedAugmenter(data_loader=dl_tr_labeled, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_train_u = NonDetMultiThreadedAugmenter(data_loader=dl_tr_unlabeled, transform=None,
                                                        num_processes=max(1, allowed_num_processes // 2),
                                                        num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # let's get this party started
        _ = next(mt_gen_train_l)
        _ = next(mt_gen_train_u)
        _ = next(mt_gen_val)

        self.dataloader_train = CombinedSemiSupervisedDataloader(
                                                        dataloader_labeled=mt_gen_train_l,
                                                        dataloader_unlabeled=mt_gen_train_u
                                                    )

        self.dataloader_val = mt_gen_val

        return self.dataloader_train, self.dataloader_val


    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:

        label_manager = plans_manager.get_label_manager(dataset_json)

        model = nnUNetTrainerNoDeepSupervision.build_network_architecture(configuration_manager.network_arch_class_name,
                                                                        configuration_manager.network_arch_init_kwargs,
                                                                        configuration_manager.network_arch_init_kwargs_req_import,
                                                                        num_input_channels,
                                                                        label_manager.num_segmentation_heads,
                                                                        enable_deep_supervision)
        return model
    

    def _compute_uncertainty(self, unlabeled_data, num_classes):
        """
        Estimate epistemic uncertainty using T stochastic forward passes through 
        the teacher (EMA) network.
        
        From UAMT paper: We perform T forward passes with input noise perturbation,
        average the predictions, and compute the predictive entropy as the 
        uncertainty measure.
        
        Args:
            unlabeled_data: Unlabeled input tensor. Shape: (B, C, D, H, W)
            num_classes: Number of segmentation classes.
            
        Returns:
            uncertainty: Per-voxel uncertainty map. Shape: (B, 1, D, H, W)
        """
        T = self.T
        batch_size = unlabeled_data.shape[0]
        spatial_dims = unlabeled_data.shape[2:]  # (D, H, W) or (H, W)
        
        # Repeat the volume T times by doing T//2 forward passes on doubled batch
        volume_batch_r = unlabeled_data.repeat(2, 1, *([1] * len(spatial_dims)))
        stride = volume_batch_r.shape[0] // 2  # = batch_size
        
        preds_shape = [stride * T, num_classes] + list(spatial_dims)
        preds = torch.zeros(preds_shape, device=unlabeled_data.device)
        
        for i in range(T // 2):
            # Add noise perturbation
            ema_inputs = volume_batch_r + torch.clamp(
                torch.randn_like(volume_batch_r) * self.noise_scale,
                -self.noise_clamp, self.noise_clamp
            )
            with torch.no_grad():
                preds[2 * stride * i: 2 * stride * (i + 1)] = self.ema_network(ema_inputs)
        
        # Softmax and compute mean prediction
        preds = torch.softmax(preds, dim=1)
        preds = preds.reshape(T, stride, num_classes, *spatial_dims)
        preds = torch.mean(preds, dim=0)  # (B, C, D, H, W)
        
        # Compute predictive entropy as uncertainty
        uncertainty = -1.0 * torch.sum(
            preds * torch.log(preds + 1e-6), dim=1, keepdim=True
        )  # (B, 1, D, H, W)
        
        return uncertainty

    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        
        # 1. Determine actual labeled batch size dynamically
        total_batch_size = data.shape[0]
        actual_labeled_bs = total_batch_size // 2
        
        data = data.to(self.device, non_blocking=True)
        
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            target_labeled = [i[:actual_labeled_bs] for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
            target_labeled = target[:actual_labeled_bs]

        self.optimizer.zero_grad(set_to_none=True)

        # 2. Split data
        data_labeled = data[:actual_labeled_bs]
        data_unlabeled = data[actual_labeled_bs:]
        
        # Determine number of classes from network output
        num_classes = self.label_manager.num_segmentation_heads

        with autocast(self.device.type, enabled=True) if (self.device.type == 'cuda' and self.use_autocast) else dummy_context():
            # --- Student forward on full batch (labeled + unlabeled) ---
            data_all = torch.cat([data_labeled, data_unlabeled], dim=0)
            student_output = self.network(data_all)
            student_output_labeled = student_output[:actual_labeled_bs]
            student_output_unlabeled = student_output[actual_labeled_bs:]
            student_output_soft = torch.softmax(student_output, dim=1)
            
            # --- Teacher forward on noisy unlabeled data ---
            noise = torch.clamp(
                torch.randn_like(data_unlabeled) * self.noise_scale,
                -self.noise_clamp, self.noise_clamp
            )
            ema_inputs = data_unlabeled + noise
            
            with torch.no_grad():
                ema_output = self.ema_network(ema_inputs)
            
            # --- Uncertainty estimation via multiple stochastic forward passes ---
            with torch.no_grad():
                uncertainty = self._compute_uncertainty(data_unlabeled, num_classes)
            
            # --- Supervised loss (only on labeled data) ---
            # Use nnUNet's built-in loss (CE + Dice)
            l_sup = self.loss(student_output_labeled, target_labeled)
            
            # --- Consistency loss with uncertainty masking ---
            consistency_weight = self.get_current_consistency_weight(
                start_epoch=0,
                end_epoch=self.num_epochs,
                max_weight=0.1
            )
            
            # MSE between student and teacher softmax outputs on unlabeled data
            consistency_dist = softmax_mse_loss(
                student_output_unlabeled, ema_output
            )  # (B, C, D, H, W)
            
            # Uncertainty threshold ramp-up
            threshold = self.get_uncertainty_threshold(num_classes=num_classes)
            
            # Create uncertainty mask: keep voxels with low uncertainty
            mask = (uncertainty < threshold).float()  # (B, 1, D, H, W)
            
            # Masked consistency loss with class-aware normalization (generalizes binary 2*sum(mask))
            consistency_loss = torch.sum(mask * consistency_dist) / (num_classes * torch.sum(mask) + 1e-16)

            # Total loss
            loss = l_sup + consistency_weight * consistency_loss

        # Backpropagation (student only)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        # Update teacher (EMA) network
        self.update_ema_variables(self.network, self.ema_network, self.ema_decay, self.global_step)
        self.global_step += 1

        return {
            'loss': loss.detach().cpu().numpy(),
            'consistency_weight': consistency_weight,
            'loss_ml': consistency_loss.detach().cpu().numpy(),
            'weighted_loss_ml': (consistency_weight * consistency_loss).detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, train_outputs: list):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)

        # Log UAMT-specific metrics
        self.logger.log('consistency_weight', float(np.mean(outputs['consistency_weight'])), self.current_epoch)
        self.logger.log('loss_ml', float(np.mean(outputs['loss_ml'])), self.current_epoch)
        self.logger.log('weighted_loss_ml', float(np.mean(outputs['weighted_loss_ml'])), self.current_epoch)
        self.logger.log('loss_cls', 0.0, self.current_epoch)  # no classification loss in UAMT

    def validation_step(self, batch: dict) -> dict:
        """Validation uses the student network only (same as CPS uses network1)."""
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        output = self.network(data)

        del data
        l = self.loss(output, target)

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    
    def save_checkpoint(self, filename: str) -> None:
        """Save student network weights (used for inference)."""
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                    'global_step': self.global_step,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')


    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file('Consistency weight (cw_ml):', 
                       np.round(self.logger.my_fantastic_logging['consistency_weight'][-1], decimals=4))
        self.print_to_log_file('Weighted loss ML:', 
                       np.round(self.logger.my_fantastic_logging['weighted_loss_ml'][-1], decimals=4))
        self.print_to_log_file('Loss ML:', 
                       np.round(self.logger.my_fantastic_logging['loss_ml'][-1], decimals=4))

        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    
    def set_deep_supervision_enabled(self, enabled: bool):
        pass


    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            _, _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []

            for i, k in enumerate(dataset_val.identifiers):
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, _, seg_prev, properties = dataset_val.load_case(k)
                data = data[:]

                if self.is_cascaded:
                    seg_prev = seg_prev[:]
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg_prev, self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()

                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )

                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)
                        dataset_class = infer_dataset_class(expected_preprocessed_folder)

                        try:
                            tmp = dataset_class(expected_preprocessed_folder, [k])
                            d, _, _, _ = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file_truncated = join(output_folder, k)

                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file_truncated, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json,
                                 default_num_processes,
                                 dataset_class),
                            )
                        ))
                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True,
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_ddp else default_num_processes)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)

        compute_gaussian.cache_clear()
