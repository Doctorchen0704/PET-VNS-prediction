"""Portable FreeSurfer command recipes and source-derived sGTM extraction.

Command builders perform no execution or writes. Use a separate processing copy
and review commands and geometry before running licensed external software.
This module does not perform identity, clinical eligibility, or visual QC.
"""
from __future__ import annotations

from pathlib import Path
import re
from collections.abc import Mapping

import numpy as np


def _arg(value):
    text = str(value)
    if not text or any(c in text for c in ("\0", "\n", "\r")):
        raise ValueError("A nonempty single argument is required")
    return text


def _subject(subject):
    if not isinstance(subject, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", subject):
        raise ValueError("Use a simple de-identified subject label")
    return subject


def _threads(threads):
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")
    return str(threads)


def recon_all_command(t1, subject, *, threads=8, cw256=False):
    """Original reconstruction options; SUBJECTS_DIR/FS_LICENSE are caller-managed.

    Never run over an existing subject directory. cw256 is an explicit technical
    choice made before outcome access, not automatically selected by this code.
    """
    command = ["recon-all", "-i", _arg(t1), "-s", _subject(subject), "-all"]
    if cw256:
        command += ["-cw256"]
    return command + ["-parallel", "-openmp", _threads(threads)]


def registration_command(pet, t1, brainmask, lta, params, final_cost, *, threads=4):
    """Six-DOF native PET-to-T1 recipe from the executed batch implementation."""
    return ["mri_coreg", "--mov", _arg(pet), "--ref", _arg(t1),
            "--ref-mask", _arg(brainmask), "--reg", _arg(lta), "--dof", "6",
            "--threads", _threads(threads), "--seed", "20260731", "--sat", "99.9",
            "--params", _arg(params), "--final-cost", _arg(final_cost)]


def analysis_grid_commands(pet, t1, brainmask, segmentation, accepted_lta, output_dir):
    """Four exact command recipes; native PET is sampled directly once to 2 mm.

    output_dir must be newly created by the caller; this function creates nothing.
    The accepted transform/segmentation must have passed technical and visual QC.
    """
    out = Path(_arg(output_dir))
    reference, pet_grid = out / "reference_T1w.nii.gz", out / "directRigid_pet.nii.gz"
    labels, mask = out / "aparcAseg_dseg.nii.gz", out / "brain_mask.nii.gz"
    return [
        ["mri_convert", _arg(t1), str(reference), "--voxsize", "2", "2", "2", "--resample_type", "cubic"],
        ["mri_vol2vol", "--mov", _arg(pet), "--targ", str(reference), "--lta", _arg(accepted_lta), "--o", str(pet_grid), "--trilin"],
        ["mri_vol2vol", "--mov", _arg(segmentation), "--targ", str(reference), "--regheader", "--o", str(labels), "--nearest"],
        ["mri_vol2vol", "--mov", _arg(brainmask), "--targ", str(reference), "--regheader", "--o", str(mask), "--nearest"],
    ]


def gtmseg_command(subject, temporary_dir):
    """Run only in a processing-copy SUBJECTS_DIR, never original study outputs."""
    return ["gtmseg", "--s", _subject(subject), "--usf", "2", "--no-seg-stats",
            "--tmpdir", _arg(temporary_dir)]


def pvc_geometry_commands(aligned_pet, conformed_t1, gtmseg_lta, subject,
                          pet_to_anat_lta, seg_to_pet_lta):
    """Header-RAS identity is valid only for already T1-aligned PET; check below."""
    return [
        ["lta_convert", "--inlta", "identity.nofile", "--src", _arg(aligned_pet),
         "--trg", _arg(conformed_t1), "--subject", _subject(subject), "--outlta", _arg(pet_to_anat_lta)],
        ["mri_concatenate_lta", "-invert1", "-invert2", "-out_type", "0",
         _arg(gtmseg_lta), _arg(pet_to_anat_lta), _arg(seg_to_pet_lta)],
    ]


def pvc_command(aligned_pet, gtmseg, color_table, pet_to_anat_lta, output_dir,
                *, fwhm_mm=5.4, threads=4):
    """mri_gtmpvc recipe; None means matched-segmentation no-PVC comparator.

    5.4 mm is the paper's principal PVC parameter. The 4 and 6 mm branches are
    retained for the historical complete processing family, not best-AUC selection.
    """
    if fwhm_mm is not None and (not np.isfinite(fwhm_mm) or fwhm_mm <= 0):
        raise ValueError("FWHM must be positive or None for matched no-PVC")
    command = ["mri_gtmpvc", "--i", _arg(aligned_pet), "--seg", _arg(gtmseg),
               "--ctab", _arg(color_table), "--reg", _arg(pet_to_anat_lta),
               "--default-seg-merge", "--no-rescale", "--auto-mask", "1", ".01",
               "--threads", _threads(threads), "--o", _arg(output_dir), "--save-yhat"]
    return command + (["--no-pvc"] if fwhm_mm is None else ["--psf", str(fwhm_mm)])


def parse_lta(text):
    """Parse the original one-transform 4x4 LTA representation without file I/O."""
    lines = text.splitlines()
    markers = [i for i, line in enumerate(lines) if line.strip() == "1 4 4"]
    kinds = [line for line in lines if line.startswith("type")]
    if len(markers) != 1 or len(kinds) != 1:
        raise ValueError("Exactly one LTA matrix and type declaration required")
    start = markers[0]
    matrix = np.asarray([[float(x) for x in line.split()] for line in lines[start + 1:start + 5]])
    kind = int(kinds[0].split("=")[1].split("#")[0])
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Finite 4x4 matrix required")
    return kind, matrix


def validate_pvc_geometry(pet_affine, anat_affine, seg_affine, pet_to_anat,
                          seg_to_pet, anatomical_to_seg):
    """Source geometry equations/thresholds. Passing is not visual registration QC."""
    pi, ai, gi = [np.asarray(a, dtype=float) for a in (pet_affine, anat_affine, seg_affine)]
    if any(a.shape != (4, 4) or not np.isfinite(a).all() for a in (pi, ai, gi)):
        raise ValueError("Finite image affines required")
    kind, ras = pet_to_anat
    k2, vox = seg_to_pet
    kg, gmat = anatomical_to_seg
    ras, vox, gmat = [np.asarray(a, dtype=float) for a in (ras, vox, gmat)]
    if any(a.shape != (4, 4) or not np.isfinite(a).all() for a in (ras, vox, gmat)):
        raise ValueError("Finite LTA matrices required")
    err = float(np.max(np.abs(vox - np.linalg.inv(pi) @ gi)))
    expected = np.linalg.inv(gi) @ ai if kg == 0 else np.eye(4)
    gerr = float(np.max(np.abs(gmat - expected)))
    if not (kind == 1 and k2 == 0 and kg in (0, 1)
            and np.allclose(ras, np.eye(4), atol=1e-6, rtol=0) and err < 1e-4 and gerr < 1e-4):
        raise ValueError("PVC header geometry does not satisfy the executed source checks")
    return {"voxel_error": err, "gtm_lta_error": gerr, "status": "HEADER_GEOMETRY_EQUIVALENCE_PASS"}


def parse_gtm_stats(text):
    """Executed eight-column gtm.stats.dat format: label column 2, value column 7."""
    rows = [s.split() for s in text.splitlines() if s.strip()]
    if not rows or any(len(r) != 8 for r in rows):
        raise ValueError("Expected nonempty eight-column gtm.stats.dat")
    ids = [int(r[1]) for r in rows]
    values = np.asarray([float(r[6]) for r in rows])
    if len(ids) != len(set(ids)) or not np.isfinite(values).all():
        raise ValueError("Unique labels and finite coefficients required")
    return ids, values


def segmentation_volumes(labels, voxel_sizes):
    labels, voxel_sizes = np.asarray(labels), np.asarray(voxel_sizes, dtype=float)
    if labels.ndim != 3 or not np.isfinite(labels).all() or not np.array_equal(labels, np.rint(labels)):
        raise ValueError("Finite three-dimensional integer segmentation required")
    if voxel_sizes.shape != (3,) or not np.isfinite(voxel_sizes).all() or np.any(voxel_sizes <= 0):
        raise ValueError("Three positive voxel sizes required")
    ids, counts = np.unique(labels, return_counts=True)
    return {int(k): float(v * np.prod(voxel_sizes)) for k, v in zip(ids, counts)}


def sgtm_features(label_ids, coefficients, volumes: Mapping[int, float], config,
                  *, xtx, xty, rounded_stats_values=None):
    """Full 84-label volume-weighted reference and six composite calculations.

    coefficients must be the unrounded gtm.nii.gz vector, in stats-file label
    order. Negative values are flagged and retained, never clipped. Individual
    regional coefficients returned here are private when actual data are used.
    """
    ids = list(label_ids)
    beta = np.asarray(coefficients, dtype=float).reshape(-1)
    if (len(ids) != len(set(ids)) or len(beta) != len(ids)
            or any(not isinstance(k, (int, np.integer)) for k in ids) or not np.isfinite(beta).all()):
        raise ValueError("Unique integer labels aligned with finite coefficients required")
    if rounded_stats_values is not None:
        stats_values = np.asarray(rounded_stats_values, dtype=float)
        if stats_values.shape != beta.shape or not np.allclose(beta, stats_values, atol=.001, rtol=1e-6):
            raise ValueError("Stats text and unrounded coefficient image disagree")
    pairs, pathways = config["bilateral_pairs"], config["primary_features"]
    if len(pathways) != 6 or any(not names or any(k not in pairs for k in names) for names in pathways.values()):
        raise ValueError("Six nonempty composites with known constituent nodes required")
    refids = sorted({int(v[h]) for v in pairs.values() for h in ("left", "right")})
    if len(refids) != 84:
        raise ValueError("Executed sGTM reference requires all 84 labels (42 bilateral nodes)")
    values = dict(zip(ids, beta))
    if any(k not in values or k not in volumes or not np.isfinite(volumes[k]) or volumes[k] <= 0 for k in refids):
        raise ValueError("Every reference label needs a coefficient and positive segmentation volume")
    xtx, xty = np.asarray(xtx, dtype=float), np.asarray(xty, dtype=float).reshape(-1)
    if xtx.shape != (len(beta), len(beta)) or xty.shape != beta.shape or not np.isfinite(xtx).all() or not np.isfinite(xty).all():
        raise ValueError("Finite full normal-equation matrices required")
    error = float(np.linalg.norm(np.linalg.solve(xtx, xty) - beta) / max(np.linalg.norm(beta), 1e-15))
    cond = float(np.linalg.cond(xtx))
    ref = float(sum(values[k] * volumes[k] for k in refids) / sum(volumes[k] for k in refids))
    if not np.isfinite(ref) or ref <= 0:
        raise ValueError("Positive finite reference required")
    nodes = {n: float(sum(values[p[h]] * volumes[p[h]] for h in ("left", "right")) /
                      sum(volumes[p[h]] for h in ("left", "right")) / ref) for n, p in pairs.items()}
    paths = {n: float(np.mean([nodes[k] for k in names])) for n, names in pathways.items()}
    bad = [k for k in refids if values[k] <= 0]
    return {"reference_raw": ref, "nodes_normalized": nodes, "pathways_normalized": paths,
            "nonpositive_reference_gm_labels": bad, "independent_solver_relative_L2_error": error,
            "condition_XtX_numpy": cond,
            "numerical_feature_flag": "ATTENTION" if bad or cond > 1e8 or error > 1e-4 else "PASS_NUMERICAL_CHECKS",
            "region_coefficients": {str(k): float(v) for k, v in values.items()}}


def extract_sgtm_files(stats_path, coefficients_path, segmentation_path, xtx_path, xty_path, config):
    """Explicit paths only; no directory discovery, writes, or clinical data joins."""
    import nibabel as nib
    from scipy.io import loadmat
    ids, rounded = parse_gtm_stats(Path(stats_path).read_text(encoding="utf-8"))
    beta = np.asarray(nib.load(coefficients_path).dataobj).reshape(-1).astype(float)
    seg = nib.load(segmentation_path)
    volumes = segmentation_volumes(np.asarray(seg.dataobj), seg.header.get_zooms()[:3])
    return sgtm_features(ids, beta, volumes, config,
                         xtx=loadmat(xtx_path)["XtX"], xty=loadmat(xty_path)["Xty"],
                         rounded_stats_values=rounded)
