#!/usr/bin/env python3
"""Convert AMBL DICOM to NIfTI, splitting multi-phase data into separate cases.

Each post-contrast phase becomes an independent case:
    output_dir/
        AMBL-001_phase1/
            AMBL-001_phase1_0000.nii.gz  (pre-contrast)
            AMBL-001_phase1_0001.nii.gz  (post-contrast phase 1)
        AMBL-001_phase2/
            AMBL-001_phase2_0000.nii.gz  (pre-contrast, same)
            AMBL-001_phase2_0001.nii.gz  (post-contrast phase 2)
    seg_dir/
        AMBL-001_phase1.nii.gz
        AMBL-001_phase2.nii.gz
"""
import os
import argparse
from collections import defaultdict

import numpy as np
import pydicom
import nibabel as nib


def dcm_folder_to_nifti(folder_path):
    """Convert a folder of DICOM slices to 3D array + affine."""
    dcm_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.dcm')])
    if not dcm_files:
        return None, None, None

    slices = []
    for f in dcm_files:
        slices.append(pydicom.dcmread(os.path.join(folder_path, f)))

    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except (AttributeError, TypeError):
        slices.sort(key=lambda s: int(s.InstanceNumber))

    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0)

    ds = slices[0]
    ps = [float(x) for x in getattr(ds, 'PixelSpacing', [1.0, 1.0])]
    st = float(getattr(ds, 'SliceThickness', 2.0))
    ipp = [float(x) for x in getattr(ds, 'ImagePositionPatient', [0, 0, 0])]

    affine = np.eye(4)
    affine[0, 0] = ps[0]
    affine[1, 1] = ps[1]
    affine[2, 2] = st
    affine[0, 3] = ipp[0]
    affine[1, 3] = ipp[1]
    affine[2, 3] = ipp[2]

    return volume, affine, ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--seg_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.seg_dir, exist_ok=True)

    base = args.input_dir
    folders = sorted([f for f in os.listdir(base) if os.path.isdir(os.path.join(base, f))])

    # Group by patient
    patient_series = defaultdict(list)
    for folder in folders:
        path = os.path.join(base, folder)
        dcm_files = [f for f in os.listdir(path) if f.endswith('.dcm')]
        if not dcm_files:
            continue
        try:
            ds = pydicom.dcmread(os.path.join(path, dcm_files[0]), stop_before_pixels=True)
            pid = getattr(ds, 'PatientID', 'Unknown')
            desc = getattr(ds, 'SeriesDescription', 'Unknown')
            patient_series[pid].append({'folder': folder, 'desc': desc, 'n_slices': len(dcm_files)})
        except:
            pass

    print(f'Found {len(patient_series)} patients')
    total_cases = 0

    for pid in sorted(patient_series.keys()):
        series_list = patient_series[pid]

        # Find pre-contrast (MASK)
        pre_series = [s for s in series_list if 'MASK' in s['desc'] and 'ROI' not in s['desc']]
        # Find ROI (segmentation)
        roi_series = [s for s in series_list if 'ROI' in s['desc']]

        if not pre_series:
            continue

        # Convert pre-contrast
        pre_path = os.path.join(base, pre_series[0]['folder'])
        pre_vol, pre_affine, _ = dcm_folder_to_nifti(pre_path)
        if pre_vol is None:
            continue
        n_slices = pre_vol.shape[0]

        # Convert ROI and determine number of phases
        seg_vol = None
        n_phases = 1
        if roi_series:
            roi_path = os.path.join(base, roi_series[0]['folder'])
            seg_vol, seg_affine, _ = dcm_folder_to_nifti(roi_path)
            if seg_vol is not None:
                seg_total = seg_vol.shape[0]
                n_phases = max(1, seg_total // n_slices)

        # Find post-contrast series (prefer Registered, single-phase +C)
        post_series = [s for s in series_list if '+C' in s['desc'] and 'Multi' not in s['desc']]
        post_registered = [s for s in post_series if 'Registered' in s['desc']]
        if post_registered:
            post_series = post_registered

        # Also check MultiPhase (can split into phases)
        multi_series = [s for s in series_list if 'MultiPhase' in s['desc']]
        multi_registered = [s for s in multi_series if 'Registered' in s['desc']]
        if multi_registered:
            multi_series = multi_registered

        # Strategy: use MultiPhase if available (more phases), else single +C
        if multi_series:
            multi_path = os.path.join(base, multi_series[0]['folder'])
            multi_vol, multi_affine, _ = dcm_folder_to_nifti(multi_path)
            if multi_vol is not None:
                multi_n_phases = multi_vol.shape[0] // n_slices
                for phase_idx in range(min(multi_n_phases, n_phases)):
                    case_name = f'{pid}_phase{phase_idx+1}'
                    case_dir = os.path.join(args.output_dir, case_name)
                    os.makedirs(case_dir, exist_ok=True)

                    # Save pre
                    nib.save(nib.Nifti1Image(pre_vol, pre_affine),
                             os.path.join(case_dir, f'{case_name}_0000.nii.gz'))
                    # Save post phase
                    start = phase_idx * n_slices
                    post_phase = multi_vol[start:start + n_slices]
                    nib.save(nib.Nifti1Image(post_phase, multi_affine),
                             os.path.join(case_dir, f'{case_name}_0001.nii.gz'))
                    # Save seg phase
                    if seg_vol is not None:
                        seg_phase = seg_vol[start:start + n_slices]
                        nib.save(nib.Nifti1Image((seg_phase > 0).astype(np.uint8), pre_affine),
                                 os.path.join(args.seg_dir, f'{case_name}.nii.gz'))

                    total_cases += 1
                continue

        # Fallback: use single +C series
        if post_series:
            for phase_idx in range(n_phases):
                post_path = os.path.join(base, post_series[min(phase_idx, len(post_series)-1)]['folder'])
                post_vol, post_affine, _ = dcm_folder_to_nifti(post_path)
                if post_vol is None:
                    continue

                case_name = f'{pid}_phase{phase_idx+1}'
                case_dir = os.path.join(args.output_dir, case_name)
                os.makedirs(case_dir, exist_ok=True)

                nib.save(nib.Nifti1Image(pre_vol, pre_affine),
                         os.path.join(case_dir, f'{case_name}_0000.nii.gz'))
                nib.save(nib.Nifti1Image(post_vol, post_affine),
                         os.path.join(case_dir, f'{case_name}_0001.nii.gz'))

                if seg_vol is not None:
                    start = phase_idx * n_slices
                    end = start + n_slices
                    if end <= seg_vol.shape[0]:
                        seg_phase = seg_vol[start:end]
                    else:
                        seg_phase = seg_vol[:n_slices]
                    nib.save(nib.Nifti1Image((seg_phase > 0).astype(np.uint8), pre_affine),
                             os.path.join(args.seg_dir, f'{case_name}.nii.gz'))

                total_cases += 1

        if total_cases % 20 == 0 and total_cases > 0:
            print(f'  {total_cases} cases created...')

    print(f'Done: {total_cases} total cases from {len(patient_series)} patients')


if __name__ == '__main__':
    main()
