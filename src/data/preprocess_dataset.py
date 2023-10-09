"""Preprocessing functions for MRI images."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Union

import torch
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_atlas_aal, fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from omegaconf import OmegaConf


def fetch_atlases(data_dir: Union[Path, str]) -> Tuple[NiftiLabelsMasker, NiftiLabelsMasker]:  # noqa: E501 BLK100
    """Fetch functional and anatomical atlases with nilean.

    Args:
        data_dir: string of folder where atlas will be stored.

    Returns:
        Tuple of (funcional_atlas, anatomical_atlas)

    Raises:
        Exception: Some error during atlas processing.
    """
    try:
        atlas_func = fetch_atlas_schaefer_2018(data_dir=data_dir)
        atlas_anat = fetch_atlas_aal(data_dir=data_dir)
        atlas_func_filename = atlas_func.maps
        atlas_anat_filename = atlas_anat.maps
    except Exception:
        raise Exception("Error occured when loaden atlases.")

    return atlas_func_filename, atlas_anat_filename


def set_conn_measure(kind: str = "covariance") -> ConnectivityMeasure:
    """Set Connectivity measure.

    Args:
        kind: string for connectivity measure.

    Returns:
        ConnectivityMeasure
    """
    return ConnectivityMeasure(kind=kind)


def nilearn_warning(subject: str, warning, mritype: str, funcname: str):
    """Treat Warning.

    Args:
        subject: string of subject id.
        warning: warning to handle.
        mritype: string of mri type (func or anat).
        funcname: string of nilearn function/method to treat.
    """
    warning.warn("Warning occured in nilearn function")


def bids_to_tensor(
    bids_folder: str,
    destination_folder: str,
    masker_func,
    masker_anat,
    corr_measure,
) -> None:
    """Read nii files from bids folder and write as torch tensor.

    Args:
        bids_folder: String of to BIDS folder location.
        destination_folder: String of directory to write into.
        masker_func: Masker object for functional connectivity.
        masker_anat: Masker object for anatomical connectivity.
        corr_measure: ConnectivityMeasuer object to calculate correlation.
    """
    subjects = os.listdir(bids_folder)
    subjects = [s for s in subjects if s.startswith("sub-")]
    for subject in subjects:
        subject_dir = os.path.join(bids_folder, subject)
        for session in os.listdir(subject_dir):
            session_dir = os.path.join(subject_dir, session)
            for mri_type in os.listdir(session_dir):
                mri_dir = os.path.join(session_dir, mri_type)
                if mri_type == "func":
                    for file in os.listdir(mri_dir):
                        if file.endswith("task-rest_bold.nii.gz"):
                            # Get the functional MRI data
                            func_img = os.path.join(
                                session_dir,
                                mri_type,
                                f"{subject}_{session}_task-rest_bold.nii.gz",
                            )
                            # mask and extract
                            try:
                                roi_ts = masker_func.fit_transform(func_img)
                            except Warning as w:
                                nilearn_warning(
                                    subject=subject,
                                    warning=w,
                                    mritype="func",
                                    funcname="masker.fit_transform",
                                )
                            try:
                                corr = corr_measure.fit_transform([roi_ts])[0]  # noqa: E501
                            except Warning as w:
                                nilearn_warning(
                                    subject=subject,
                                    warning=w,
                                    mritype="func",
                                    funcname="corr_measure.fit_transform",
                                )
                            corr = torch.from_numpy(
                                corr,
                            )
                            save_file = os.path.join(
                                destination_folder,
                                f"{mri_type}_{subject}_{session}.pt",
                            )
                            torch.save(corr, str(save_file))
                if mri_type == "anat":
                    for file in os.listdir(mri_dir):
                        if file.endswith("T1w_defacemask.nii.gz"):
                            # get anatomical MRI data
                            anat_img = os.path.join(
                                session_dir,
                                mri_type,
                                f"{subject}_{session}_T1w_defacemask.nii.gz",
                            )  # noqa: E501
                            # mask and extract
                            try:
                                roi_ts = masker_anat.fit_transform(anat_img)
                            except Warning as w:
                                nilearn_warning(
                                    subject=subject,
                                    warning=w,
                                    mritype="anat",
                                    funcname="masker.fit_transform",
                                )
                            try:
                                corr = corr_measure.fit_transform([roi_ts])[0]
                            except Warning as w:
                                nilearn_warning(
                                    subject=subject,
                                    warning=w,
                                    mritype="anat",
                                    funcname="corr_measure.fit_transform",
                                )
                            corr = torch.from_numpy(
                                corr,
                            )
                            save_file = os.path.join(
                                destination_folder,
                                f"{mri_type}_{subject}_{session}.pt",
                            )
                            torch.save(corr, str(save_file))


def preprocess_connmats(source_dir: Union[str, Path],
                        destination_dir: Union[str, Path] = "",
                        thrshld: float = 0.3) -> None:
    """Preprocess connectivity matrices.

    Connectiviy measures that are in absolute value below threshold
    will be set to zero.

    Args:
        source_dir: string of directory containing conn matrices in .pt format.
        destination_dir: string of directory to save processed data.
        thrshld: float of threshold. If correlation below, then set to 0.
    """
    conmatrices = os.listdir(source_dir)
    for connectome in conmatrices:
        mat = torch.load(os.path.join(source_dir, connectome))
        mat_masked = torch.where(  # noqa BLK 100
            torch.abs(mat) > thrshld, torch.tensor(0.0), mat)  # noqa BLK 100
        if destination_dir == "":
            torch.save(mat_masked, Path(destination_dir, connectome))
        else:
            torch.save(mat_masked, Path(source_dir, connectome))


if __name__ == "__main__":
    config = OmegaConf.load("configs/data_preprocess.yaml")
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = Path(project_dir, "data")

    masker_func, masker_anat = fetch_atlases(
        Path(data_dir, "external"))  # writes atlas into external
    connectivity_measure = config.get("mriprep").get(
        "parcellation").get("connectivity_measure")
    conn = set_conn_measure(kind=connectivity_measure)
    # create pt files of mri images
    # bids_to_tensor(
    #    bids_folder=Path(data_dir, "raw", "ds004169"),
    #    destination_folder=Path(data_dir, "interim"),
    #    masker_func=masker_func,
    #    masker_anat=masker_anat
    # )

    # mask conn matrices
    thrshld = config.get("mriprep").get("preprocessing").get("threshold")
    preprocess_connmats(
        source_dir=Path(data_dir, "interim"),
        destination_dir=Path(data_dir, "preprocessed"),
        thrshld=thrshld
    )
