"""
nnUNetTrainer_DCNET: DCNet implementation for nnUNet framework.
DCNet uses one encoder and two decoders with:
- Attention-based knowledge distillation loss between encoder and decoder features
- Distance-based confidence learning (loss_dc0)
- Cross-entropy on pseudo labels (loss_cer)
- Adaptive threshold-based masking for confidence selection
"""

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


class Attention(nn.Module):
    """
    Attention transfer loss for knowledge distillation.
    Computes attention maps and measures MSE between student and teacher attention maps.
    """
    def __init__(self, p=2):
        super(Attention, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        """
        Compute attention loss between student and teacher features.
        Args:
            g_s: List of student feature maps (encoder features)
            g_t: List of teacher feature maps (decoder features)
        Returns:
            Attention transfer loss
        """
        loss = sum([self.at_loss(f_s, f_t.detach()) for f_s, f_t in zip(g_s, g_t)])
        return loss

    def at_loss(self, f_s, f_t):
        """Compute attention transfer loss between two feature maps."""
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        """Compute normalized attention map from feature map."""
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))


class DualDecoderWrapperDCNet(nn.Module):
    """
    DualDecoderWrapper for DCNet.
    Wraps two decoders around a single encoder and returns:
    - Two segmentation outputs (output1, output2)
    - Encoder features for attention loss
    - Decoder features from both decoders for attention loss
    """
    def __init__(self, encoder, decoder, n_classes):
        super().__init__()
        self.encoder = encoder
        self.decoder1 = copy.deepcopy(decoder)
        self.decoder2 = copy.deepcopy(decoder)
        self.inference_mode = False  # When True, only run encoder+decoder1 and return tensor

        self.seg_head1 = nn.Conv3d(n_classes, n_classes, kernel_size=1)
        self.seg_head2 = nn.Conv3d(n_classes, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass - get all skip connections
        skips = self.encoder(x)
        
        # Extract encoder features for attention loss (excluding the bottleneck)
        # skips[0] is highest resolution, skips[-1] is bottleneck
        encoder_features = skips[:-1]  # Take all except bottleneck

        # Decoder 1 forward pass
        f1_outputs = self.decoder1(skips)
        features1 = f1_outputs[0] if isinstance(f1_outputs, list) else f1_outputs
        output1 = self.seg_head1(features1)

        # In inference mode, only use encoder + decoder1 and return a tensor directly
        if self.inference_mode:
            return output1

        # Decoder 2 forward pass
        f2_outputs = self.decoder2(skips)
        features2 = f2_outputs[0] if isinstance(f2_outputs, list) else f2_outputs
        output2 = self.seg_head2(features2)

        # Extract decoder features for attention loss
        # These are intermediate decoder feature maps at different resolutions
        if isinstance(f1_outputs, list):
            decoder_features1 = f1_outputs
        else:
            decoder_features1 = [f1_outputs]
            
        if isinstance(f2_outputs, list):
            decoder_features2 = f2_outputs
        else:
            decoder_features2 = [f2_outputs]

        out = {
            "output1": output1,
            "output2": output2,
            "encoder_features": encoder_features,
            "decoder_features1": decoder_features1,
            "decoder_features2": decoder_features2,
        }
        return out


class CombinedSemiSupervisedDataloader:
    """Combines labeled and unlabeled dataloaders for semi-supervised learning."""
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

        # Combine batches: [Labeled, Unlabeled]
        combined = {
            "data": torch.cat([batch_l["data"], batch_u["data"]], dim=0),
            "target": torch.cat([batch_l["target"], batch_u["target"]], dim=0)
        }
        return combined


class nnUNetTrainerDCNET(nnUNetTrainerNoDeepSupervision):
    """
    Trainer for DCNet: Dual-decoder Consistency Network for semi-supervised learning.
    
    Key components:
    1. One encoder, two decoders architecture
    2. Attention-based knowledge distillation loss (loss_at_kd)
    3. Distance-based confidence loss (loss_dc0)
    4. Cross-entropy on pseudo labels (loss_cer)
    5. Adaptive threshold-based masking
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
            seed_balance: str = '42'
        ):
        super().__init__(plans, configuration, fold, dataset_json, device, percentage_labeled_data, seed_name, seed_balance)

        # Toggle mixed-precision (autocast) for DCNet training
        self.use_autocast = True

        # NNunet base settings
        self.percentage_labeled_data = percentage_labeled_data
        self.seed_name = seed_name
        self.seed_balance = seed_balance
        
        print(f"DCNet Trainer: Percentage of labeled data: {self.percentage_labeled_data}")
        
        ### Hyperparameters
        self.initial_lr = 1e-2  # 0.01
        self.weight_decay = 3e-5  # 0.00003
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 1000
        self.current_epoch = 0
        self.enable_deep_supervision = False

        # DCNet specific hyperparameters
        self.consistency = 0.1  # Max consistency weight
        self.consistency_rampup = 200.0  # Epochs for ramp-up
        self.temperature = 0.5  # Temperature for sharpening
        self.cer_weight = 0.3  # Weight for cross-entropy on pseudo labels
        self.at_kd_scale = 1000.0  # Scale for attention loss
        self.dc0_scale = 1000.0  # Scale for distance-based confidence loss

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_criterion = F.mse_loss
        self.criterion_att = Attention(p=2)

        # Threshold for confidence selection (starts at 1/num_classes)
        self.cur_threshold = None

        # Iteration counter for threshold adaptation
        self.iter_num = 0

        self.logger = nnUNetLoggerCBS()
        
        # Initialize DCNet-specific logging keys
        self.logger.my_fantastic_logging['loss_seg_dice'] = list()
        self.logger.my_fantastic_logging['loss_at_kd'] = list()
        self.logger.my_fantastic_logging['loss_dc0'] = list()
        self.logger.my_fantastic_logging['loss_cer'] = list()
        self.logger.my_fantastic_logging['cur_threshold'] = list()
        self.logger.my_fantastic_logging['mean_max_values'] = list()

    def get_current_consistency_weight(self, start_epoch=0, end_epoch=1000, max_weight=0.1):
        """
        Sigmoid ramp-up of consistency weight.
        Gradually increases from 0 to max_weight between start_epoch and end_epoch.
        """
        def sigmoid_rampup(current, rampup_length):
            if rampup_length == 0:
                return 1.0
            else:
                current = np.clip(current, 0.0, rampup_length) # Clips (limits) the current epoch value to be within [0, rampup_length].
                phase = 1.0 - current / rampup_length # Normalizes the current epoch into a "ramp phase" between 1 and 0.
                return float(np.exp(-5.0 * phase * phase))

        rampup_length = end_epoch - start_epoch
        current = self.current_epoch - start_epoch

        if self.current_epoch < start_epoch:
            return 0.0
        elif self.current_epoch >= end_epoch:
            return max_weight
        else:
            return max_weight * sigmoid_rampup(current, rampup_length)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True
        )
        
        lr_scheduler = PolyLRScheduler(
            optimizer,
            self.initial_lr,
            self.num_epochs
        )
        return optimizer, lr_scheduler

    def initialize(self):
        if not self.was_initialized:
            self._set_batch_size_and_oversample()
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json
            )
            self.network = self.build_network_architecture(
                self.plans_manager,
                self.dataset_json,
                self.configuration_manager,
                self.num_input_channels,
                enable_deep_supervision=False
            ).to(self.device)
            
            # Compile network for speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            # DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            # Initialize threshold based on number of classes
            num_classes = self.label_manager.num_segmentation_heads
            self.cur_threshold = 1.0 / num_classes

            self.was_initialized = True
        else:
            raise RuntimeError("Trainer already initialized.")

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def do_split(self):
        """Split data into labeled, unlabeled, and validation sets."""
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
            dataset = self.dataset_class(
                self.preprocessed_dataset_folder,
                identifiers=None,
                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
            )
            
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

                # Return early for 100% labeled
                if self.percentage_labeled_data == 1.0:
                    self.print_to_log_file(
                        "Using 100% labeled data: falling back to default nnU-Net split."
                    )
                    tr_keys_limit = tr_keys
                    return tr_keys_limit, val_keys
            else:
                self.print_to_log_file(
                    "INFO: Requested fold %d but splits contain only %d folds. "
                    "Creating random 80:20 split!" % (self.fold, len(splits))
                )
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.identifiers))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file(
                    "Random 80:20 split: %d training, %d validation cases."
                    % (len(tr_keys), len(val_keys))
                )

            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file(
                    'WARNING: Some validation cases are also in training set.'
                )

            parent_dir = dirname(nnUNet_preprocessed)
            split_file = join(
                parent_dir,
                "all_splits_trainset",
                f"balanced_labeled_splits_fold_{self.seed_name}.json"
            )

            with open(split_file, "r") as f:
                balanced_splits = json.load(f)
                print(f"Using balanced splits from {split_file}")

            # Percentage selector for semi-supervised case
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

            self.print_to_log_file(
                f"Using {len(labeled_cases)} labeled cases ({percentage_key}, balanced) from {split_file}"
            )

            tr_keys_limit = labeled_cases
            tr_unlabeled_keys = unlabeled_cases

        self.print_to_log_file(
            f"Using {len(labeled_cases)} labeled training cases out of {len(tr_keys)} total. "
            f"{len(unlabeled_cases)} unlabeled. {len(val_keys)} validation."
        )

        return tr_keys_limit, tr_unlabeled_keys, val_keys

    def get_tr_and_val_datasets(self):
        tr_labeled_keys, tr_unlabeled_keys, val_keys = self.do_split()

        dataset_tr_labeled = self.dataset_class(
            self.preprocessed_dataset_folder, tr_labeled_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )
        dataset_tr_unlabeled = self.dataset_class(
            self.preprocessed_dataset_folder, tr_unlabeled_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )
        dataset_val = self.dataset_class(
            self.preprocessed_dataset_folder, val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )
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

        # Training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label
        )

        # Validation pipeline
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label
        )

        dataset_tr_labeled_split, dataset_tr_unlabeled_split, dataset_val_split = self.get_tr_and_val_datasets()

        # DATALOADERS
        dl_tr_labeled = nnUNetDataLoader(
            dataset_tr_labeled_split, self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        )

        dl_tr_unlabeled = nnUNetDataLoader(
            dataset_tr_unlabeled_split, self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        )

        dl_val = nnUNetDataLoader(
            dataset_val_split, self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train_l = SingleThreadedAugmenter(dl_tr_labeled, None)
            mt_gen_train_u = SingleThreadedAugmenter(dl_tr_unlabeled, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train_l = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr_labeled, transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002
            )
            mt_gen_train_u = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr_unlabeled, transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002
            )
        
        # Trigger initial batch loading
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
    def build_network_architecture(
            plans_manager: PlansManager,
            dataset_json,
            configuration_manager: ConfigurationManager,
            num_input_channels,
            enable_deep_supervision: bool = False
    ) -> nn.Module:

        label_manager = plans_manager.get_label_manager(dataset_json)

        model = nnUNetTrainerNoDeepSupervision.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            label_manager.num_segmentation_heads,
            enable_deep_supervision
        )
        
        # Wrap encoder and decoder in DCNet dual-decoder wrapper
        network = DualDecoderWrapperDCNet(
            encoder=model.encoder,
            decoder=model.decoder,
            n_classes=label_manager.num_segmentation_heads
        )
        return network

    def train_step(self, batch: dict) -> dict:
        """
        DCNet training step implementing:
        1. Supervised dice loss on labeled data
        2. Attention-based knowledge distillation loss (loss_at_kd)
        3. Distance-based confidence loss (loss_dc0)
        4. Cross-entropy on pseudo labels (loss_cer)
        """
        data = batch['data']
        target = batch['target']

        # Determine labeled batch size (batch is [Labeled, Unlabeled] concatenated)
        total_batch_size = data.shape[0]
        labeled_bs = total_batch_size // 2

        data = data.to(self.device, non_blocking=True)

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        num_classes = self.label_manager.num_segmentation_heads
        max_iterations = self.num_epochs * self.num_iterations_per_epoch

        with autocast(self.device.type, enabled=True) if (self.device.type == 'cuda' and self.use_autocast) else dummy_context():
            # Forward pass
            output = self.network(data)

            output1 = output['output1']
            output2 = output['output2']
            encoder_features = output['encoder_features']
            decoder_features1 = output['decoder_features1']
            decoder_features2 = output['decoder_features2']

            # Softmax outputs
            output1_soft = F.softmax(output1, dim=1)
            output2_soft = F.softmax(output2, dim=1)

            # Sharpened softmax outputs (temperature = 0.5)
            output1_soft0 = F.softmax(output1 / self.temperature, dim=1)
            output2_soft0 = F.softmax(output2 / self.temperature, dim=1)

            # ============================================================
            # Threshold calculation (adaptive threshold based on confidence)
            # ============================================================
            with torch.no_grad():
                max_values1, _ = torch.max(output1_soft, dim=1)
                max_values2, _ = torch.max(output2_soft, dim=1)
                
                # Progress percentage
                percent = (self.iter_num + 1) / max_iterations

                cur_threshold1 = (1 - percent) * self.cur_threshold + percent * max_values1.mean()
                cur_threshold2 = (1 - percent) * self.cur_threshold + percent * max_values2.mean()
                mean_max_values = min(max_values1.mean(), max_values2.mean())

                self.cur_threshold = min(cur_threshold1, cur_threshold2)
                self.cur_threshold = torch.clip(self.cur_threshold, 0.25, 0.95)

            # ============================================================
            # Mask generation for confidence-based selection
            # ============================================================
            # High confidence mask: both outputs agree with high confidence
            mask_high = (output1_soft > self.cur_threshold) & (output2_soft > self.cur_threshold)
            mask_non_similarity = (mask_high == False)

            # Masked outputs for distance-based learning
            new_output1_soft = torch.mul(mask_non_similarity, output1_soft)
            new_output2_soft = torch.mul(mask_non_similarity, output2_soft)
            
            # High confidence outputs
            high_output1 = torch.mul(mask_high, output1)
            high_output2 = torch.mul(mask_high, output2)
            high_output1_soft = torch.mul(mask_high, output1_soft)
            high_output2_soft = torch.mul(mask_high, output2_soft)

            # Pseudo labels
            pseudo_output1 = torch.argmax(output1_soft, dim=1)
            pseudo_output2 = torch.argmax(output2_soft, dim=1)
            pseudo_high_output1 = torch.argmax(high_output1_soft, dim=1)
            pseudo_high_output2 = torch.argmax(high_output2_soft, dim=1)

            # ============================================================
            # Distance-based confidence learning (loss_dc0)
            # ============================================================
            # Find pixels where one decoder is more confident than the other
            max_output1_indices = new_output1_soft > new_output2_soft
            max_output2_indices = new_output2_soft > new_output1_soft

            max_output1_value0 = torch.mul(max_output1_indices, output1_soft0)
            min_output2_value0 = torch.mul(max_output1_indices, output2_soft0)
            max_output2_value0 = torch.mul(max_output2_indices, output2_soft0)
            min_output1_value0 = torch.mul(max_output2_indices, output1_soft0)

            loss_dc0 = 0.0
            loss_dc0 += self.mse_criterion(max_output1_value0.detach(), min_output2_value0)
            loss_dc0 += self.mse_criterion(max_output2_value0.detach(), min_output1_value0)

            # ============================================================
            # Attention-based knowledge distillation loss (loss_at_kd)
            # ============================================================
            # Match encoder features with decoder2 features for knowledge transfer
            # Align the number of feature maps (take min of lengths)
            min_len = min(len(encoder_features), len(decoder_features2))
            encoder_feat_aligned = encoder_features[:min_len]
            decoder2_feat_aligned = decoder_features2[:min_len]
            
            loss_at_kd = self.criterion_att(encoder_feat_aligned, decoder2_feat_aligned)

            # ============================================================
            # Supervised segmentation loss (Dice loss on labeled data)
            # ============================================================
            loss_seg_dice = 0.0
            loss_seg_dice += self.loss(output1[:labeled_bs], target[:labeled_bs])
            loss_seg_dice += self.loss(output2[:labeled_bs], target[:labeled_bs])

            # ============================================================
            # Cross-entropy on pseudo labels (loss_cer)
            # ============================================================
            loss_cer = 0.0
            if mean_max_values >= 0.95:
                # High confidence: use full pseudo labels
                loss_cer += self.ce_loss(output1, pseudo_output2.long().detach())
                loss_cer += self.ce_loss(output2, pseudo_output1.long().detach())
            else:
                # Lower confidence: only use high-confidence regions
                loss_cer += self.ce_loss(high_output1, pseudo_high_output2.long().detach())
                loss_cer += self.ce_loss(high_output2, pseudo_high_output1.long().detach())

            # ============================================================
            # Total loss combination (following DCNet formulation)
            # ============================================================
            ssl_weight = self.get_current_consistency_weight(
                start_epoch=0,
                end_epoch=self.num_epochs,
                max_weight=0.1
            )
            
            # loss = supervised_loss + (1-cw) * at_kd + cw * dc0 + 0.3 * cer
            supervised_loss = loss_seg_dice
            loss = (supervised_loss + 
                    (1 - ssl_weight) * (self.at_kd_scale * loss_at_kd) + 
                    ssl_weight * (self.dc0_scale * loss_dc0) + 
                    self.cer_weight * loss_cer)

        # Backward pass with gradient scaling (mixed precision)
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

        self.iter_num += 1

        return {
            'loss': loss.detach().cpu().numpy(),
            'loss_seg_dice': loss_seg_dice.detach().cpu().numpy() if torch.is_tensor(loss_seg_dice) else float(loss_seg_dice),
            'loss_at_kd': loss_at_kd.detach().cpu().numpy() if torch.is_tensor(loss_at_kd) else float(loss_at_kd),
            'loss_dc0': loss_dc0.detach().cpu().numpy() if torch.is_tensor(loss_dc0) else float(loss_dc0),
            'loss_cer': loss_cer.detach().cpu().numpy() if torch.is_tensor(loss_cer) else float(loss_cer),
            'cur_threshold': float(self.cur_threshold) if torch.is_tensor(self.cur_threshold) else self.cur_threshold,
            'mean_max_values': float(mean_max_values) if torch.is_tensor(mean_max_values) else mean_max_values,
            'consistency_weight': ssl_weight,
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

        # Log DCNet-specific metrics
        self.logger.log('loss_seg_dice', float(np.mean(outputs['loss_seg_dice'])), self.current_epoch)
        self.logger.log('loss_at_kd', float(np.mean(outputs['loss_at_kd'])), self.current_epoch)
        self.logger.log('loss_dc0', float(np.mean(outputs['loss_dc0'])), self.current_epoch)
        self.logger.log('loss_cer', float(np.mean(outputs['loss_cer'])), self.current_epoch)
        self.logger.log('cur_threshold', float(np.mean(outputs['cur_threshold'])), self.current_epoch)
        self.logger.log('mean_max_values', float(np.mean(outputs['mean_max_values'])), self.current_epoch)
        self.logger.log('consistency_weight', float(np.mean(outputs['consistency_weight'])), self.current_epoch)

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if (self.device.type == 'cuda' and self.use_autocast) else dummy_context():
            # Set inference mode: only encoder + decoder1 → returns tensor directly
            mod = self.network
            if isinstance(mod, DDP):
                mod = mod.module
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod
            mod.inference_mode = True

            output1 = self.network(data)

            mod.inference_mode = False

        del data
        l = self.loss(output1, target)

        axes = [0] + list(range(2, output1.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output1) > 0.5).long()
        else:
            output_seg = output1.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output1.shape, device=output1.device, dtype=torch.float32)
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
        """Save checkpoint with model weights."""
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
                    'iter_num': self.iter_num,  # DCNet specific
                    'cur_threshold': float(self.cur_threshold) if torch.is_tensor(self.cur_threshold) else self.cur_threshold,  # DCNet specific
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
        
        # DCNet specific logging
        self.print_to_log_file('loss_seg_dice:', 
                       np.round(self.logger.my_fantastic_logging['loss_seg_dice'][-1], decimals=4))
        self.print_to_log_file('loss_at_kd:', 
                       np.round(self.logger.my_fantastic_logging['loss_at_kd'][-1], decimals=6))
        self.print_to_log_file('loss_dc0:', 
                       np.round(self.logger.my_fantastic_logging['loss_dc0'][-1], decimals=6))
        self.print_to_log_file('loss_cer:', 
                       np.round(self.logger.my_fantastic_logging['loss_cer'][-1], decimals=4))
        self.print_to_log_file('cur_threshold:', 
                       np.round(self.logger.my_fantastic_logging['cur_threshold'][-1], decimals=4))
        self.print_to_log_file('mean_max_values:', 
                       np.round(self.logger.my_fantastic_logging['mean_max_values'][-1], decimals=4))
        self.print_to_log_file('consistency_weight:', 
                       np.round(self.logger.my_fantastic_logging['consistency_weight'][-1], decimals=4))

        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # Periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # Handle 'best' checkpointing
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

        # Enable inference mode so DualDecoderWrapperDCNet returns tensor (encoder+decoder1 only)
        mod = self.network
        if isinstance(mod, DDP):
            mod = mod.module
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.inference_mode = True

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file(
                "WARNING! batch size is 1 during training and torch.compile is enabled. "
                "If you encounter crashes in validation then this is because torch.compile "
                "forgets to trigger a recompilation of the model with deep supervision disabled."
            )

        predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
            perform_everything_on_device=True, device=self.device, verbose=False,
            verbose_preprocessing=False, allow_tqdm=False
        )
        predictor.manual_initialization(
            self.network, self.plans_manager, self.configuration_manager, None,
            self.dataset_json, self.__class__.__name__,
            self.inference_allowed_mirroring_axes
        )

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            _, _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = self.dataset_class(
                self.preprocessed_dataset_folder, val_keys,
                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
            )

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
                                f"Predicting next stage {n} failed for case {k} because "
                                f"the preprocessed file is missing!"
                            )
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
            metrics = compute_metrics_on_folder(
                join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                validation_output_folder,
                join(validation_output_folder, 'summary.json'),
                self.plans_manager.image_reader_writer_class(),
                self.dataset_json["file_ending"],
                self.label_manager.foreground_regions if self.label_manager.has_regions else
                self.label_manager.foreground_labels,
                self.label_manager.ignore_label, chill=True,
                num_processes=default_num_processes * dist.get_world_size() if
                self.is_ddp else default_num_processes
            )
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)

        compute_gaussian.cache_clear()

        # Reset inference mode back to training (full dual decoder) mode
        mod.inference_mode = False
