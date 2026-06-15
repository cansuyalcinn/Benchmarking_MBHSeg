from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch


class nnUNetTrainerNoDeepSupervision(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
        percentage_labeled_data: float = 1.0, 
        seed_name: str = '0', 
        seed_balance = '42'
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, percentage_labeled_data, seed_name, seed_balance)
        self.enable_deep_supervision = False
