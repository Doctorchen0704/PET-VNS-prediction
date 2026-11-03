"""Check the explicit LTA equations, then optionally execute one regional sGTM fit."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step_support import require_review, run_commands
from pet_vns_prediction.preprocessing import parse_lta, validate_pvc_geometry, pvc_command


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subject', required=True)
    ap.add_argument('--aligned-pet', type=Path, required=True)
    ap.add_argument('--conformed-t1', type=Path, required=True, help='The same processing-copy mri/nu.mgz used to make pet-to-anat LTA')
    ap.add_argument('--gtmseg', type=Path, required=True)
    ap.add_argument('--color-table', type=Path, required=True)
    ap.add_argument('--pet-to-anat-lta', type=Path, required=True)
    ap.add_argument('--seg-to-pet-lta', type=Path, required=True)
    ap.add_argument('--gtmseg-lta', type=Path, required=True)
    ap.add_argument('--fwhm-mm', type=float, choices=[4.0,5.4,6.0], default=5.4)
    ap.add_argument('--matched-no-pvc', action='store_true', help='Same segmentation comparator, not the original primary PET branch')
    ap.add_argument('--qc-table', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='Fresh parent directory; fit outputs go to fit/')
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    require_review(a.qc_table, a.subject, 'registration_review')
    require_review(a.qc_table, a.subject, 'coverage_review')
    require_review(a.qc_table, a.subject, 'pvc_segmentation_review')
    import nibabel as nib
    pet, anat, seg = [nib.load(str(p)) for p in (a.aligned_pet, a.conformed_t1, a.gtmseg)]
    transforms = [parse_lta(p.read_text(encoding='utf-8')) for p in (a.pet_to_anat_lta, a.seg_to_pet_lta, a.gtmseg_lta)]
    check = validate_pvc_geometry(pet.affine, anat.affine, seg.affine, *transforms)
    print(json.dumps(check))
    if a.output.exists():
        raise FileExistsError('Use a new PVC fit directory')
    if not a.color_table.is_file():
        raise FileNotFoundError(a.color_table)
    command = pvc_command(a.aligned_pet.resolve(), a.gtmseg.resolve(), a.color_table.resolve(),
        a.pet_to_anat_lta.resolve(), a.output.resolve()/'fit',
        fwhm_mm=None if a.matched_no_pvc else a.fwhm_mm, threads=4)
    print(json.dumps(run_commands([command], a.output, a.execute)))


if __name__ == '__main__':
    main()
