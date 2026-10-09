"""Source-derived PET and structural MRI feature calculations.

These functions start with aligned image arrays or existing FreeSurfer stats;
they do not run segmentation, registration, PVC, or clinical QC. See
``docs/preprocessing.md`` for the boundary between this public interface and the
study's image-processing workflow. No participant information is embedded here.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def load_feature_config(path: str | Path) -> dict[str, Any]:
    """Read an explicitly supplied, non-patient feature-definition JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_arrays(pet: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pet, labels = np.asarray(pet), np.asarray(labels)
    if pet.ndim != 3 or pet.shape != labels.shape:
        raise ValueError("PET and labels must be matching three-dimensional arrays")
    if not np.all(np.isfinite(labels)) or not np.array_equal(labels, np.rint(labels)):
        raise ValueError("Labels must be finite integers")
    return pet, labels.astype(np.int32, copy=False)


def normalization_masks(
    pet: np.ndarray, labels: np.ndarray, brainmask: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Construct the study reference masks on an already aligned image grid.

    Finite zero-valued PET voxels remain in these masks. This reproduces the
    reference-mean rule; ROI extraction uses a different, strictly-positive rule.
    """
    pet, labels = _validate_arrays(pet, labels)
    brainmask = np.asarray(brainmask)
    if brainmask.shape != pet.shape:
        raise ValueError("Brain mask and PET shapes differ")
    if not np.all(np.isfinite(brainmask)):
        raise ValueError("Brain mask must be finite")
    definition = config["normalization"]
    brain = brainmask > 0
    finite = np.isfinite(pet)
    low, high = definition["cortical_label_range_half_open"]
    cortical = (labels >= low) & (labels < high)
    subcortical = np.isin(labels, definition["supratentorial_subcortical_labels"])
    return {
        "supraGM": (cortical | subcortical) & brain & finite,
        "wholeBrain": brain & finite,
        "cerebellarCortex": np.isin(labels, definition["cerebellar_cortex_labels"]) & brain & finite,
    }


def normalize_pet(
    pet: np.ndarray, labels: np.ndarray, brainmask: np.ndarray,
    reference: str, config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Divide PET by a reference mean, returning float32 as in saved study images.

    This is not SUV conversion or a complete QC gate. The returned counts and
    flags must be inspected alongside registration and segmentation QC. Inputs
    are never modified. Negative/nonfinite inputs are rejected: they failed the
    original image-level hard QC even though finite masks could be constructed.
    """
    pet, labels = _validate_arrays(pet, labels)
    if not np.all(np.isfinite(pet)) or np.any(pet < 0):
        raise ValueError("Study image-level QC requires finite, nonnegative PET")
    masks = normalization_masks(pet, labels, brainmask, config)
    if reference not in masks:
        raise ValueError(f"Unknown normalization reference: {reference}")
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    means = {name: float(np.mean(pet[mask])) if mask.any() else float("nan")
             for name, mask in masks.items()}
    if not np.isfinite(means[reference]) or means[reference] <= 0:
        raise ValueError("Selected reference is empty or has a nonpositive mean")
    brain = np.asarray(brainmask) > 0
    positive_fraction = float(np.mean(pet[brain] > 0)) if brain.any() else 0.0
    minimums = config["normalization"]["image_qc_minimum_reference_voxels"]
    all_references_pass = (
        all(np.isfinite(value) and value > 0 for value in means.values())
        and all(counts[name] >= minimums[name] for name in masks)
        and positive_fraction >= 0.95
    )
    return (pet / means[reference]).astype(np.float32), {
        "reference": reference,
        "reference_means": means,
        "reference_voxel_counts": counts,
        "brain_positive_coverage_fraction": positive_fraction,
        "reference_intensity_gate_pass": bool(all_references_pass),
        "geometry_and_visual_qc": "not_evaluated_by_this_function",
    }


def roi_stats(
    pet: np.ndarray,
    mask: np.ndarray,
    hard_label_minimum: int,
    attention_label_maximum: int,
    valid_pass_minimum: float,
    valid_attention_minimum: float,
) -> dict[str, Any]:
    """Study ROI summary; retain missing/attention states instead of imputing."""
    label_voxels = int(np.count_nonzero(mask))
    values = pet[mask]
    valid = np.isfinite(values) & (values > 0)
    valid_voxels = int(np.count_nonzero(valid))
    valid_fraction = valid_voxels / label_voxels if label_voxels else 0.0

    if label_voxels < hard_label_minimum:
        status = "missing_label_below_hard_minimum"
        mean = None
        median = None
    elif valid_fraction < valid_attention_minimum:
        status = "missing_insufficient_valid_pet_fraction"
        mean = None
        median = None
    else:
        selected = values[valid].astype(np.float64, copy=False)
        mean = float(np.mean(selected))
        median = float(np.median(selected))
        if label_voxels <= attention_label_maximum:
            status = "attention_low_label_voxels"
        elif valid_fraction < valid_pass_minimum:
            status = "attention_valid_pet_fraction"
        else:
            status = "pass"

    return {
        "label_voxels": label_voxels,
        "valid_pet_voxels": valid_voxels,
        "valid_pet_fraction": valid_fraction,
        "mean": mean,
        "median": median,
        "status": status,
    }


def worst_status(statuses: list[str]) -> str:
    if any(status == "missing" or status.startswith("missing_") for status in statuses):
        return "missing"
    if any(status == "attention" or status.startswith("attention_") for status in statuses):
        return "attention"
    return "pass"


def extract_pet_features(
    normalized_pet: np.ndarray, labels: np.ndarray, config: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract 19 bilateral node means and six unweighted node composites.

    The PET must already be normalized using one chosen reference. Left/right
    voxels are pooled, NOT averaged as two equally weighted hemisphere means.
    Different nodes then receive equal weight within a composite.
    """
    pet, labels = _validate_arrays(normalized_pet, labels)
    rules = config["voxel_and_missingness_rules"]
    arguments = (
        int(rules["minimum_label_voxels_hard"]),
        int(rules["label_voxels_attention_range"][1]),
        float(rules["valid_pet_fraction_pass_minimum"]),
        float(rules["valid_pet_fraction_attention_minimum"]),
    )
    pair_results: dict[str, Any] = {}
    for name, pair in config["bilateral_pairs"].items():
        left_id, right_id = int(pair["left"]), int(pair["right"])
        left = roi_stats(pet, labels == left_id, *arguments)
        right = roi_stats(pet, labels == right_id, *arguments)
        bilateral = roi_stats(pet, (labels == left_id) | (labels == right_id), *arguments)
        if left["mean"] is None or right["mean"] is None or (left["mean"] + right["mean"]) == 0:
            asymmetry = None
        else:
            asymmetry = 2.0 * (left["mean"] - right["mean"]) / (left["mean"] + right["mean"])
        pair_results[name] = {
            "label_ids": {"left": left_id, "right": right_id},
            "class": pair["class"], "left": left, "right": right, "bilateral": bilateral,
            "signed_asymmetry_index": asymmetry,
            "pair_qc_status": worst_status([left["status"], right["status"], bilateral["status"]]),
        }
    primary_features: dict[str, Any] = {}
    for feature_name, node_names in config["primary_features"].items():
        node_values = [pair_results[node]["bilateral"]["mean"] for node in node_names]
        node_statuses = [pair_results[node]["pair_qc_status"] for node in node_names]
        if any(value is None for value in node_values):
            value = None
            status = "missing"
        else:
            value = float(np.mean(np.asarray(node_values, dtype=np.float64)))
            status = worst_status(node_statuses)
        primary_features[feature_name] = {
            "value": value, "status": status, "constituent_nodes": list(node_names),
            "constituent_bilateral_means": dict(zip(node_names, node_values, strict=True)),
        }
    return {"pair_results": pair_results, "primary_features": primary_features,
            "overall_primary_feature_qc_status": worst_status(
                [item["status"] for item in primary_features.values()])}


def extract_pet_nifti(
    normalized_pet_path: str | Path, labels_path: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Explicit-path adapter; requires nibabel and the T1-aligned 2-mm grid.

    Loading and canonical orientation do not perform registration/resampling.
    Returned values contain no input filenames or participant identifiers.
    """
    import nibabel as nib

    pet_image = nib.as_closest_canonical(nib.load(str(normalized_pet_path)))
    label_image = nib.as_closest_canonical(nib.load(str(labels_path)))
    if pet_image.shape != label_image.shape or not np.allclose(
        pet_image.affine, label_image.affine, atol=1e-5
    ):
        raise ValueError("PET and segmentation geometry differ")
    if not np.allclose(pet_image.header.get_zooms()[:3], [2, 2, 2], rtol=0, atol=1e-4):
        raise ValueError("Study feature extraction requires a 2-mm isotropic grid")
    return extract_pet_features(np.asanyarray(pet_image.dataobj), np.asanyarray(label_image.dataobj), config)


def parse_aseg(path: str | Path) -> tuple[float, dict[str, float]]:
    """Read eTIV and structure volumes, not header identity information."""
    etiv: float | None = None
    volumes: dict[str, float] = {}
    etiv_pattern = re.compile(
        r"^# Measure EstimatedTotalIntraCranialVol,\s*eTIV,.*?,\s*"
        r"([0-9.+\-eE]+),\s*mm\^3\s*$"
    )
    with Path(path).open("r", encoding="utf-8", errors="strict") as handle:
        for raw in handle:
            line = raw.strip()
            match = etiv_pattern.match(line)
            if match:
                etiv = float(match.group(1))
                continue
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    volumes[parts[4]] = float(parts[3])
                except ValueError:
                    continue
    if etiv is None or not math.isfinite(etiv) or etiv <= 0:
        raise ValueError("Invalid or missing eTIV")
    return etiv, volumes


def parse_aparc(path: str | Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with Path(path).open("r", encoding="utf-8", errors="strict") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                rows[parts[0]] = {
                    "surface_area_mm2": float(parts[2]),
                    "thickness_mm": float(parts[4]),
                }
            except ValueError:
                continue
    return rows


def extract_t1_features(
    etiv_mm3: float, aseg_volumes: Mapping[str, float],
    aparc: Mapping[str, Mapping[str, Mapping[str, float]]], config: Mapping[str, Any],
) -> dict[str, float]:
    """Six source-defined FreeSurfer features, without cohort-specific gates."""
    if not math.isfinite(etiv_mm3) or etiv_mm3 <= 0:
        raise ValueError("eTIV must be finite and positive")
    features: dict[str, float] = {}
    for feature in config["t1_features"]:
        name, measure = feature["name"], feature["measure"]
        if measure == "normalized_volume":
            structures = feature["aseg_structures"]
            missing = [label for label in structures if label not in aseg_volumes]
            if missing:
                raise ValueError(f"{name}: missing required aseg structures {missing}")
            values = [aseg_volumes[label] for label in structures]
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f"{name}: nonpositive/nonfinite volume")
            features[name] = sum(values) / etiv_mm3
        elif measure == "area_weighted_thickness_mm":
            weighted_sum, total_area = 0.0, 0.0
            for hemi in ("lh", "rh"):
                if hemi not in aparc:
                    raise ValueError(f"Missing hemisphere: {hemi}")
                for label in feature["aparc_parcels"]:
                    if label not in aparc[hemi]:
                        raise ValueError(f"{name}: missing {hemi} parcel {label}")
                    item = aparc[hemi][label]
                    area, thickness = item["surface_area_mm2"], item["thickness_mm"]
                    if not math.isfinite(area) or not math.isfinite(thickness) or area <= 0 or thickness <= 0:
                        raise ValueError(f"{name}: invalid {hemi} {label}")
                    weighted_sum += area * thickness
                    total_area += area
            features[name] = weighted_sum / total_area
        else:
            raise ValueError(f"Unsupported measure: {measure}")
    return features


def extract_t1_stats(
    aseg_path: str | Path, left_aparc_path: str | Path,
    right_aparc_path: str | Path, config: Mapping[str, Any],
) -> dict[str, float]:
    """Read three explicitly supplied FreeSurfer stats files; no writes."""
    etiv, aseg = parse_aseg(aseg_path)
    aparc = {"lh": parse_aparc(left_aparc_path), "rh": parse_aparc(right_aparc_path)}
    return extract_t1_features(etiv, aseg, aparc, config)
