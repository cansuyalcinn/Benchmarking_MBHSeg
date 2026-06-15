"""
Testing Script — Run nnUNet inference on the test set.
Loads a trained model checkpoint and predicts segmentations for all images in imagesTs.
Run evaluation_test.py afterwards to compute metrics.

Usage:
    python testing.py --dataset Dataset400_Mbhseg25 --model nnUNetTrainerCPS__nnUNetPlans__3d_fullres__perc0.2_seed0_seedbalance42 --fold 0
"""
import os
import sys
import argparse
import torch

# Ensure our custom nnUNet is importable
sys.path.insert(0, "/home/cansu/MBHSEG25_NICVICOROB/nnUNet")

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def main():
    parser = argparse.ArgumentParser(description="Run nnUNet inference on the test set (imagesTs)")
    parser.add_argument("--dataset", "-d", type=str, required=True,
                        help="Dataset name, e.g. Dataset400_Mbhseg25")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Model folder name, e.g. nnUNetTrainerCPS__nnUNetPlans__3d_fullres__perc0.2_seed0_seedbalance42")
    parser.add_argument("--fold", "-f", type=int, default=0,
                        help="Fold number (default: 0)")
    parser.add_argument("--checkpoint", "-chk", type=str, default="checkpoint_best.pth",
                        help="Checkpoint filename (default: checkpoint_best.pth)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"],
                        help="Device to run inference on (default: cuda)")
    parser.add_argument("--disable_tta", action="store_true", default=False,
                        help="Disable test-time augmentation (mirroring). Faster but less accurate.")
    parser.add_argument("--step_size", type=float, default=0.5,
                        help="Step size for sliding window (default: 0.5)")
    parser.add_argument("--save_probabilities", action="store_true", default=False,
                        help="Save softmax probabilities as npz files")
    parser.add_argument("--npp", type=int, default=3,
                        help="Number of processes for preprocessing (default: 3)")
    parser.add_argument("--nps", type=int, default=3,
                        help="Number of processes for segmentation export (default: 3)")
    parser.add_argument("--gpu", type=int, default=1,
                        help="GPU id to use if device is cuda (default: 0)")
    args = parser.parse_args()

    # -------------------------
    # Paths
    # -------------------------
    # assign GPU if using cuda
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    data_base = "/home/user/Benchmarking_MBHSeg/data/nnUNet"
    input_folder = os.path.join(data_base, "nnUNet_raw", args.dataset, "imagesTs")
    model_folder = os.path.join(data_base, "nnUNet_results", args.dataset, args.model)
    output_folder = os.path.join(model_folder, f"fold_{args.fold}", "test")

    assert os.path.isdir(input_folder), f"Input folder not found: {input_folder}"
    assert os.path.isdir(model_folder), f"Model folder not found: {model_folder}"
    os.makedirs(output_folder, exist_ok=True)

    print(f"Input images:  {input_folder}")
    print(f"Model folder:  {model_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Fold:          {args.fold}")
    print(f"TTA:           {'disabled' if args.disable_tta else 'enabled'}")

    # -------------------------
    # Device setup
    # -------------------------
    if args.device == "cpu":
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device("cpu")
    elif args.device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

    # -------------------------
    # Run inference
    # -------------------------
    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=model_folder,
        use_folds=[args.fold],
        checkpoint_name=args.checkpoint
    )

    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_folder,
        output_folder_or_list_of_truncated_output_files=output_folder,
        save_probabilities=args.save_probabilities,
        overwrite=True,
        num_processes_preprocessing=args.npp,
        num_processes_segmentation_export=args.nps,
    )

    print(f"\nDone! Predictions saved to: {output_folder}")


if __name__ == "__main__":
    main()
