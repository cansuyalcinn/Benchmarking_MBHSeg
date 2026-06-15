
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

# FPML model. 

class FeaturePerturbation(nn.Module):
    def __init__(self, kap=0.1, eps=1e-6, use_gpu=True):
        super(FeaturePerturbation, self).__init__()
        # self.num_features = num_features
        self.eps = eps
        self.kap = kap
        self.use_gpu = use_gpu

    def forward(self, x):
        # normalization
        mu = x.mean(dim=[2, 3, 4], keepdim=True)  # [B,C,1,1,1]
        var = x.var(dim=[2, 3, 4], keepdim=True)  # [B,C,1,1,1]
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        batch_mu = mu.mean(dim=[0], keepdim=True)  # [1,C,1,1,1]
        batch_psi = (mu.var(dim=[0], keepdim=True) + self.eps).sqrt()  # [1,C,1,1,1]
        batch_sig = sig.mean(dim=[0], keepdim=True)  # [1,C,1,1,1]
        # batch_mu = average of the per-sample means across the batch → global mean of means
        # batch_sig = average of per-sample stds → global mean of stds
        # batch_psi = std of mu values across the batch → diversity of means
        # batch_phi = std of sig values across the batch → diversity of stds
        batch_phi = (sig.var(dim=[0], keepdim=True) + self.eps).sqrt()  # [1,C,1,1,1]
        # epsilon is sampled from a uniform distribution in the range [-kap, kap].
        epsilon = torch.empty(1).uniform_(-self.kap, self.kap)
        # This epsilon is then used to add perturbation noise to the feature statistics
        # gamma and beta control the scaling and shifting of the normalized features (x_normed), 
        # so perturbing them directly affects the final perturbed feature map x_aug
        epsilon_tensor = torch.tensor(epsilon, device=sig.device, dtype=sig.dtype)
        #  Noise scaled by batch diversity
        gamma =  sig + epsilon_tensor * batch_phi
        beta = mu + epsilon_tensor * batch_psi
        x_aug = gamma * x_normed + beta
        return x_aug

class DualDecoderWrapper(nn.Module):
    # This class wraps two decoders around a single encoder.
    # This class also applies feature perturbation to the outputs of the decoders.

    def __init__(self, encoder, decoder, n_classes, num_FP=3):
        super().__init__()
        self.encoder = encoder
        self.decoder1 = copy.deepcopy(decoder)
        self.decoder2 = copy.deepcopy(decoder)
        self.num_FP = num_FP
        self.inference_mode = False  # When True, only run encoder+decoder1 and return tensor

        self.seg_head1 = nn.Conv3d(n_classes, n_classes, kernel_size=1)
        self.boundary_head1 = nn.Conv3d(n_classes, n_classes, kernel_size=1)

        self.seg_head2 = nn.Conv3d(n_classes, n_classes, kernel_size=1)
        self.boundary_head2 = nn.Conv3d(n_classes, n_classes, kernel_size=1)

        self.FP_module = FeaturePerturbation().cuda()

    def forward(self, x):
        skips = self.encoder(x)
        f1_outputs = self.decoder1(skips)

        features1 = f1_outputs[0] if isinstance(f1_outputs, list) else f1_outputs
        logits_d1 = self.seg_head1(features1)

        # In inference mode, only use encoder + decoder1 and return a tensor directly
        if self.inference_mode:
            return logits_d1

        f2_outputs = self.decoder2(skips)
        features2 = f2_outputs[0] if isinstance(f2_outputs, list) else f2_outputs

        logits_d2 = self.seg_head2(features2)

        # apply feature perturbation N times. 
        # we apply perturbations in feature space before the last convolution. after FP we pass them to conv layer.
        # and then we get logits for each decoder.
        logits_d1_fp = []
        logits_d2_fp = []
        for i in range(self.num_FP):
            f1_fp = self.FP_module(features1)
            f2_fp = self.FP_module(features2)
            logits_d1_fp.append(self.seg_head1(f1_fp))  # perturb → seg head
            logits_d2_fp.append(self.seg_head2(f2_fp))

        # Apply heads
        out = {
            "seg1": logits_d1,
            "seg2": logits_d2,
            "features1": features1,
            "features2": features2, 
            "features1_FP": logits_d1_fp,
            "features2_FP": logits_d2_fp
        }
        return out


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

        # print(f"Labeled batch size: {batch_l['data'].shape[0]}, Unlabeled batch size: {batch_u['data'].shape[0]}")
        # 2 and 2 = in total 4 as batch size.

        # Combine batches
        combined = {
            "data": torch.cat([batch_l["data"], batch_u["data"]], dim=0),
            "target": torch.cat([batch_l["target"], batch_u["target"]], dim=0)
        }
        combined["keys"] = list(batch_l.get("keys", [])) + list(batch_u.get("keys", []))
        return combined


class nnUNetTrainerFPMLchallenge(nnUNetTrainerNoDeepSupervision):
    """
    Trainer for semi-supervised learning with mutual learning on segmentation heads.
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

        # Toggle mixed-precision (autocast) for FPML training.
        self.use_autocast = True

        # NNunet base settings
        self.percentage_labeled_data = percentage_labeled_data
        self.seed_name = seed_name # fold name in the splits file, e.g. fold_0, fold_1, etc.
        self.seed_balance = seed_balance # seed for balanced split (0,1,42)
        self.consistency_criterion_ml = nn.CrossEntropyLoss(reduction='none')
        print(f"Percentage of labeled data: {self.percentage_labeled_data}")
        
        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-2 # 0.01
        self.weight_decay = 3e-5 # 0.00003
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 1000
        self.current_epoch = 0
        self.enable_deep_supervision = False # we disable deep supervision to have a fair comparison, since our ssl models will be based on nnunet without DS. 


        self.logger = nnUNetLoggerCBS()

        # about batch size: we use batch size 2, because it meansn that we get 2 from labeled 2 from unlabeled, and in the end we have 4 as a batch size actually. for this framework.

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
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()
            self.num_input_channels = determine_num_input_channels(self.plans_manager, 
                                                                   self.configuration_manager,
                                                                   self.dataset_json)
            self.network = self.build_network_architecture(
                self.plans_manager,
                self.dataset_json,
                self.configuration_manager,
                self.num_input_channels,
                enable_deep_supervision=False
            ).to(self.device)
            
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                                "That should not happen.")

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        from nnunetv2.paths import nnUNet_preprocessed
        from os.path import join, dirname
        import json
        
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            dataset = self.dataset_class(self.preprocessed_dataset_folder,
                                         identifiers=None,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.identifiers)))
                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
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

            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
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

            # percentage selector (only for semi-supervised case)
            if self.percentage_labeled_data == 0.1:
                percentage_key = "10_percent"
            elif self.percentage_labeled_data == 0.2:
                percentage_key = "20_percent"
            elif self.percentage_labeled_data == 0.3:
                percentage_key = "30_percent"
            elif self.percentage_labeled_data == 1.0:
                percentage_key = "100_percent"
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

            # for unlabeled cases just take the first 84 cases among them. 
            tr_unlabeled_keys = tr_unlabeled_keys[:84]

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

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr_labeled_split, dataset_tr_unlabeled_split, dataset_val_split = self.get_tr_and_val_datasets()

        # DATALOADERS
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
        # # let's get this party started
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
        # Wrap encoder and decoder
        network  = DualDecoderWrapper(
        encoder=model.encoder,
        decoder=model.decoder,
        n_classes=label_manager.num_segmentation_heads)
        return network

    def calculate_ML_FP(self, y_all, outfeats, outputs_seg, y_all_fp, i, j, num_classes, consistency_criterion):
        # Calculates Mutual Learning with Feature Perturbation.

        def entropy_map(prob):
            return -torch.sum(prob * torch.log(prob + 1e-6), dim=1)  # shape: (B, D, H, W)

        # === STEP 1: Get most confident predictions (min-entropy) for decoder i and j ===

        # For decoder i
        probs_i_all = [y_all[i]] + [y_all_fp[i][k] for k in range(len(y_all_fp[i]))]
        entropies_i_stack = torch.stack([entropy_map(p) for p in probs_i_all])  # shape: (3, B, D, H, W)
        min_entropy_i = torch.min(entropies_i_stack, dim=0)[0]  # (B, D, H, W)

        # For decoder j
        probs_j_all = [y_all[j]] + [y_all_fp[j][k] for k in range(len(y_all_fp[j]))]
        entropies_j_stack = torch.stack([entropy_map(p) for p in probs_j_all])  # shape: (3, B, D, H, W)
        min_entropy_indices_j = torch.argmin(entropies_j_stack, dim=0)  # (B, D, H, W) 
        # min_entropy_indices_j: a tensor with shape (B, D, H, W) containing the index (0, 1, or 2) of the version (original or perturbed) that has the lowest entropy for each voxel.

        # === STEP 2: Build y_all_confident_j (voxel-wise most confident prediction) ===
        stacked_probs_j = torch.stack(probs_j_all, dim=0)  # shape: (3, B, C, D, H, W)
        stacked_probs_j = stacked_probs_j.permute(1, 0, 2, 3, 4, 5)  # shape: (B, 3, C, D, H, W) Now, for each voxel, you have 3 predictions across dimension dim=1
        gather_idx = min_entropy_indices_j.unsqueeze(1)  # (B, 1, D, H, W)
        selected_probs_j = torch.gather(
            stacked_probs_j,  #stacked_probs_j: shape = (B, 3, C, D, H, W)
            dim=1,  # We want to gather along dim=1 (i.e., the 3 options)
            # The index must match the shape of stacked_probs_j except in dim=1
            index=gather_idx.unsqueeze(2).expand(-1, -1, num_classes, -1, -1, -1) # (B, 1, C, D, H, W)
        )  # shape: (B, 1, C, D, H, W). For each voxel, gather from dim=1 using the selected version index.
        # This tensor now contains, for each voxel, the softmax from the most confident version.

        y_all_confident_j = selected_probs_j.squeeze(1)  # shape: (B, C, D, H, W)

        # === STEP 3: Confidence-based mutual supervision mask ===
        min_entropy_j = torch.gather(entropies_j_stack, dim=0, index=min_entropy_indices_j.unsqueeze(0)).squeeze(0)
        mask = (min_entropy_i > min_entropy_j).float()  # decoder j is more confident → it teaches i

        # === STEP 4: Build prototype bank from y_all_confident_j ===
        batch_o, c_o, w_o, h_o, d_o = y_all[j].shape
        batch_f, c_f, w_f, h_f, d_f = outfeats[j].shape
        # Preserving original teacher features (outfeats[j]) to avoid extra compute, but it can be extended later.
        teacher_f = outfeats[j].reshape(batch_f, c_f, -1)

        index = torch.argmax(y_all_confident_j, dim=1, keepdim=True)  # use confident predictions

        prototype_bank = torch.zeros(batch_f, num_classes, c_f).cuda()
        for ba in range(batch_f):
            for n_class in range(num_classes):
                mask_temp = (index[ba] == n_class).float()
                top_fea = outfeats[j][ba] * mask_temp  # outfeats[j] is original
                prototype_bank[ba, n_class] = top_fea.sum(-1).sum(-1).sum(-1) / (mask_temp.sum() + 1e-6)

        prototype_bank = F.normalize(prototype_bank, dim=-1)

        # === STEP 5: Compute pixel-wise similarity to prototype bank ===
        mask_t = torch.zeros_like(y_all[i]).cuda()
        for ba in range(batch_o):
            for n_class in range(num_classes):
                class_prototype = prototype_bank[ba, n_class]
                mask_t[ba, n_class] = F.cosine_similarity(
                    teacher_f[ba],
                    class_prototype.unsqueeze(1),
                    dim=0
                ).view(w_f, h_f, d_f)

        weight_pixel_t = (1 - nn.MSELoss(reduction='none')(mask_t, y_all_confident_j)).mean(1)
        weight_pixel_t = weight_pixel_t * mask

        # === STEP 6: Consistency loss using confident pseudo-labels ===
        loss_t = consistency_criterion(
            outputs_seg[i],
            torch.argmax(y_all_confident_j, dim=1).detach()
        )

        return (loss_t * weight_pixel_t.detach()).sum() / (mask.sum() + 1e-6)

    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        # 1. Determine actual labeled batch size dynamically
        # The batch is concatenated [Labeled, Unlabeled].
        # Since we use the same batch_size for both loaders, the split is exactly half.
        total_batch_size = data.shape[0]
        self.labeled_bs = total_batch_size // 2

        data = data.to(self.device, non_blocking=True)

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if (self.device.type == 'cuda' and self.use_autocast) else dummy_context():
            output = self.network(data)

            output_seg1 = output['seg1']
            output_seg2 = output['seg2']
            features1 = output['features1']
            features2 = output['features2']
            features1_FP = output['features1_FP']
            features2_FP = output['features2_FP']

            outfeats = [features1, features2]
            outputs_seg = [output_seg1, output_seg2]
            outputs_seg_FP = [features1_FP, features2_FP]

            num_outputs = len(outputs_seg)  # 2 decoders
            self.num_FP = 3

            # Full segmentation loss on the labeled data
            loss_seg = 0
            for idx in range(num_outputs):
                loss_seg += self.loss(outputs_seg[idx][:self.labeled_bs], target[:self.labeled_bs])

            # Mutual learning loss
            y_all = torch.zeros((num_outputs,) + outputs_seg[0].shape).to(self.device)
            for idx in range(num_outputs):
                y = outputs_seg[idx]
                y_prob = F.softmax(y, dim=1)
                y_all[idx] = y_prob

            # Go over Feature Perturbated predictions and convert them to probabilities.
            # y_all_fp[0, i] = softmax output of the i-th perturbed version of decoder 1
            # y_all_fp[1, i] = softmax output of the i-th perturbed version of decoder 2
            # y_all_fp has shape (num_outputs, num_FP, B, C, D, H, W)
            y_all_fp = torch.zeros((num_outputs, self.num_FP) + outputs_seg[0].shape).to(self.device)
            for dec_idx in range(num_outputs):
                for fp_idx in range(self.num_FP):
                    y = outputs_seg_FP[dec_idx][fp_idx]
                    y_prob = F.softmax(y, dim=1)
                    y_all_fp[dec_idx, fp_idx] = y_prob

            loss_ml_fp = 0
            for i in range(num_outputs):
                for j in range(num_outputs):
                    if i != j:
                        loss_ml_fp += self.calculate_ML_FP(y_all=y_all, outfeats=outfeats,
                                                    outputs_seg=outputs_seg, y_all_fp=y_all_fp,
                                                    i=i, j=j, num_classes=self.label_manager.num_segmentation_heads,
                                                    consistency_criterion=self.consistency_criterion_ml)

            cw_ml = self.get_current_consistency_weight(start_epoch=0, end_epoch=1000, max_weight=0.1)
            weighted_loss_ml_fp = cw_ml * loss_ml_fp
            l = (0.5 * loss_seg) + (cw_ml * loss_ml_fp)


        # Backward pass with grad_scaler (mixed precision) or plain
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            'loss': l.detach().cpu().numpy(),
            'consistency_weight': cw_ml,
            'loss_ml': loss_ml_fp.detach().cpu().numpy() if torch.is_tensor(loss_ml_fp) else float(loss_ml_fp),
            'weighted_loss_ml': weighted_loss_ml_fp.detach().cpu().numpy() if torch.is_tensor(weighted_loss_ml_fp) else float(weighted_loss_ml_fp),
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

        # Log FPML-specific metrics
        self.logger.log('consistency_weight', float(np.mean(outputs['consistency_weight'])), self.current_epoch)
        self.logger.log('loss_ml', float(np.mean(outputs['loss_ml'])), self.current_epoch)
        self.logger.log('weighted_loss_ml', float(np.mean(outputs['weighted_loss_ml'])), self.current_epoch)

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # Autocast for mixed-precision during validation (matching train_step behavior)
        with autocast(self.device.type, enabled=True) if (self.device.type == 'cuda' and self.use_autocast) else dummy_context():
            # Set inference mode: only encoder + decoder1 → returns tensor directly
            mod = self.network
            if isinstance(mod, DDP):
                mod = mod.module
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod
            mod.inference_mode = True

            output_seg1 = self.network(data)

            mod.inference_mode = False

        del data
        l = self.loss(output_seg1, target)

        axes = [0] + list(range(2, output_seg1.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output_seg1) > 0.5).long()
        else:
            output_seg = output_seg1.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output_seg1.shape, device=output_seg1.device, dtype=torch.float32)
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
        # we will just save the model 1 weights to be used in the inference. 
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

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
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

        # Enable inference mode so DualDecoderWrapper returns tensor (encoder+decoder1 only)
        mod = self.network
        if isinstance(mod, DDP):
            mod = mod.module
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.inference_mode = True

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

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

                val_keys = val_keys[self.local_rank:: dist.get_world_size()]
                # we cannot just have barriers all over the place because the number of keys each GPU receives can be
                # different

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

                # we do [:] to convert blosc2 to numpy
                data = data[:]

                if self.is_cascaded:
                    seg_prev = seg_prev[:]
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg_prev, self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )
                # for debug purposes
                # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
                #      self.dataset_json, output_filename_truncated, save_probabilities)

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)
                        # next stage may have a different dataset class, do not use self.dataset_class
                        dataset_class = infer_dataset_class(expected_preprocessed_folder)

                        try:
                            # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
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

                        # resample_and_save(prediction, target_shape, output_file, self.plans_manager, self.configuration_manager, properties,
                        #                   self.dataset_json)
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
                # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
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

        # Reset inference mode back to training (full dual decoder) mode
        mod.inference_mode = False
