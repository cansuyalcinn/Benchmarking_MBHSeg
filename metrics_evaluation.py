"""
Test Set Evaluation — Dice, NSD@1mm, Sensitivity, Specificity per class.
Compares predictions against GT labels from labelsTs, computes per-class metrics,
and saves a per-patient CSV plus a readable text file with mean metrics.
"""
import os
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk

# Add surface-distance library to path
sys.path.insert(0, "/home/user/surface-distance")
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance


# -------------------------
# Metric functions
# -------------------------
def dice_coefficient_binary(pred, gt):
    intersection = np.sum(pred * gt)
    denominator = np.sum(pred) + np.sum(gt)
    if denominator == 0:
        return np.nan
    return 2.0 * intersection / denominator


def nsd_at_tolerance(pred, gt, spacing_mm, tolerance_mm=1.0):
    """Compute Normalized Surface Dice (NSD) at a given tolerance in mm."""
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    if not np.any(pred_bool) and not np.any(gt_bool):
        return np.nan
    if not np.any(pred_bool) or not np.any(gt_bool):
        return 0.0
    surface_distances = compute_surface_distances(gt_bool, pred_bool, spacing_mm)
    return compute_surface_dice_at_tolerance(surface_distances, tolerance_mm)


def sensitivity_binary(pred, gt):
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    tp = np.sum(pred_bool & gt_bool)
    fn = np.sum(~pred_bool & gt_bool)
    denominator = tp + fn
    if denominator == 0:
        return np.nan
    return tp / denominator


def specificity_binary(pred, gt):
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    tn = np.sum(~pred_bool & ~gt_bool)
    fp = np.sum(pred_bool & ~gt_bool)
    denominator = tn + fp
    if denominator == 0:
        return np.nan
    return tn / denominator


def metrics_per_class(pred, gt, class_labels, spacing_mm, tolerance_mm=1.0):
    scores = {}
    for c in class_labels:
        pred_c = (pred == c).astype(np.uint8)
        gt_c = (gt == c).astype(np.uint8)
        scores[f"dice_class_{c}"] = dice_coefficient_binary(pred_c, gt_c)
        scores[f"nsd1_class_{c}"] = nsd_at_tolerance(pred_c, gt_c, spacing_mm, tolerance_mm)
        scores[f"sensitivity_class_{c}"] = sensitivity_binary(pred_c, gt_c)
        scores[f"specificity_class_{c}"] = specificity_binary(pred_c, gt_c)
    return scores


class_labels = [1, 2, 3, 4, 5]
tolerance_mm = 1.0


# -------------------------
# Evaluate all test patients
# -------------------------
def main(dataset_name, model_folder, fold):
    base_path = f"/home/user/Benchmarking_MBHSeg/data/nnUNet/nnUNet_raw/{dataset_name}"
    images_ts = os.path.join(base_path, "imagesTs")
    labels_ts = os.path.join(base_path, "labelsTs")

    prediction_folder_name = "test"
    predictions_dir = f"/home/user/Benchmarking_MBHSeg/data/nnUNet/nnUNet_results/{dataset_name}/{model_folder}/fold_{fold}/{prediction_folder_name}"
    results_base = f"/home/user/Benchmarking_MBHSeg/data/nnUNet/nnUNet_results/{dataset_name}/{model_folder}/fold_{fold}"
    csv_path = os.path.join(results_base, f"{prediction_folder_name}_dice_nsd_multiclass.csv")
    mean_txt_path = os.path.join(results_base, f"{prediction_folder_name}_mean_metrics.txt")

    print(f"Predictions dir: {predictions_dir}")
    print(f"Labels dir:      {labels_ts}")
    print(f"CSV output:      {csv_path}")
    print(f"Mean TXT output: {mean_txt_path}")

    dice_cols = [f"dice_class_{c}" for c in class_labels]
    nsd_cols = [f"nsd1_class_{c}" for c in class_labels]
    sens_cols = [f"sensitivity_class_{c}" for c in class_labels]
    spec_cols = [f"specificity_class_{c}" for c in class_labels]
    metric_cols = dice_cols + nsd_cols + sens_cols + spec_cols
    columns = ["name"] + metric_cols
    results = pd.DataFrame(columns=columns)

    for file in sorted(os.listdir(images_ts)):
        if not file.endswith(".nii.gz"):
            continue

        patient_id = file.split(".")[0].split("_")[1]
        name = "mbhseg_" + patient_id

        gt_path = os.path.join(labels_ts, name + ".nii.gz")
        pred_path = os.path.join(predictions_dir, name + ".nii.gz")

        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            print(f"[SKIP] Missing GT or prediction for {name}")
            continue

        gt_sitk = sitk.ReadImage(gt_path)
        pred_sitk = sitk.ReadImage(pred_path)

        gt = sitk.GetArrayFromImage(gt_sitk)
        pred = sitk.GetArrayFromImage(pred_sitk)

        # Get voxel spacing in (z, y, x) order to match array axis order
        spacing_xyz = gt_sitk.GetSpacing()        # (x, y, z)
        spacing_mm = list(reversed(spacing_xyz))   # (z, y, x)

        if gt.shape != pred.shape:
            print(f"[SKIP] Shape mismatch for {name}: GT {gt.shape}, Pred {pred.shape}")
            continue

        scores = metrics_per_class(pred, gt, class_labels, spacing_mm, tolerance_mm)
        row = {"name": name}
        row.update(scores)
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

        dice_str = "  ".join([f"C{c}={scores[f'dice_class_{c}']:.3f}" for c in class_labels])
        nsd_str = "  ".join([f"C{c}={scores[f'nsd1_class_{c}']:.3f}" for c in class_labels])
        sens_str = "  ".join([f"C{c}={scores[f'sensitivity_class_{c}']:.3f}" for c in class_labels])
        spec_str = "  ".join([f"C{c}={scores[f'specificity_class_{c}']:.3f}" for c in class_labels])
        print(f"{name} | Dice: {dice_str} | NSD@1: {nsd_str} | Sens: {sens_str} | Spec: {spec_str}")

    # Append MEAN row
    mean_row = {"name": "MEAN"}
    for col in metric_cols:
        mean_row[col] = results[col].mean()
    results = pd.concat([results, pd.DataFrame([mean_row])], ignore_index=True)

    # Save CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results.to_csv(csv_path, index=False)

    # Save readable mean metrics summary
    with open(mean_txt_path, "w", encoding="utf-8") as f:
        f.write("Test set mean metrics (mean across patients, reported per class)\n")
        f.write(f"dataset: {dataset_name}\n")
        f.write(f"model:   {model_folder}\n")
        f.write(f"fold:    {fold}\n\n")

        f.write("Dice (mean)\n")
        for c in class_labels:
            f.write(f"  class {c}: {mean_row[f'dice_class_{c}']:.6f}\n")
        f.write("\n")

        f.write("NSD@1mm (mean)\n")
        for c in class_labels:
            f.write(f"  class {c}: {mean_row[f'nsd1_class_{c}']:.6f}\n")
        f.write("\n")

        f.write("Sensitivity (mean)\n")
        for c in class_labels:
            f.write(f"  class {c}: {mean_row[f'sensitivity_class_{c}']:.6f}\n")
        f.write("\n")

        f.write("Specificity (mean)\n")
        for c in class_labels:
            f.write(f"  class {c}: {mean_row[f'specificity_class_{c}']:.6f}\n")

    print(f"\nResults saved to: {csv_path}")
    print(f"Mean metrics saved to: {mean_txt_path}")
    print(results.tail(1).to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test set evaluation: Dice, NSD@1mm, Sensitivity, Specificity per class")
    parser.add_argument("--dataset", "-d", type=str, default="Dataset200_Mbhseg24",
                        help="Dataset name, e.g. Dataset400_Mbhseg25")
    parser.add_argument("--model", "-m", type=str, default="nnUNetTrainer__nnUNetPlans__3d_fullres__perc0.2_seed0_seedbalance0",
                        help="Model folder name")
    parser.add_argument("--fold", "-f", type=int, default=0,
                        help="Fold number (default: 0)")
    args = parser.parse_args()
    main(args.dataset, args.model, args.fold)
