import argparse
import os
import shutil
import tempfile
import sys
from copy import deepcopy
from typing import Tuple, Union, List

import numpy as np
import torch
import SimpleITK as sitk
from fastapi import APIRouter, UploadFile,FastAPI
from fastapi.responses import FileResponse
from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax, save_segmentation_nifti
from nnunet.postprocessing.connected_components import load_remove_save, load_postprocessing
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.one_hot_encoding import to_one_hot
from nnunet.inference.predict import check_input_folder_and_return_caseIDs
from src.configuration.config import *

# Conditional import for different platforms
if 'win' in sys.platform:
    import pathos
    Process = pathos.helpers.mp.Process
    Queue = pathos.helpers.mp.Queue
else:
    from multiprocessing import Process, Queue

# Define FastAPI router
router = APIRouter()

# Helper functions

def get_temp_file(input_folder: str) -> str:
    """Generate a temporary file path for storing the uploaded NIFTI file."""
    fd, niftyPath = tempfile.mkstemp(suffix=".nii.gz", dir=input_folder, prefix='tmp')
    return fd, niftyPath

def clean_up_temp_file(fd: int, niftyPath: str) -> None:
    """Clean up the temporary file after processing."""
    try:
        if fd != -1:  # Check if the file descriptor is valid
            os.close(fd)
    except OSError as e:
        print(f"Error closing file descriptor: {e}")
    
    if os.path.exists(niftyPath):
        os.remove(niftyPath)


def prepare_output_files(input_folder: str, case_ids: List[str], output_folder: str) -> Tuple[List[List[str]], List[str]]:
    """Prepare the output files for prediction."""
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [
        [join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and len(i) == (len(j) + 12)]
        for j in case_ids
    ]
    
    cleaned_output_files = []
    for case_id in case_ids:
        output_file = join(output_folder, f"{case_id}.nii.gz")
        dr, f = os.path.split(output_file)
        maybe_mkdir_p(dr)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(join(dr, f))

    return list_of_lists, cleaned_output_files

def process_prediction(trainer, list_of_lists: List[List[str]], output_files: List[str], checkpoint_params) -> None:
    """Process prediction for each patient."""
    print("---------------------------------------------------------------------")
    print("Inside process_prediction")
    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            output_file = output_files[i]
            print("Preprocessing", output_file)
            
            d, _, dct = trainer.preprocess_patient(l)
            print(output_file, dct)
            print(d.shape)
            
            # Handle large files
            if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 to be safe, 4 for float32 bytes
                print("This output is too large for Python process-process communication. Saving output temporarily to disk.")
                np.save(output_file[:-7] + ".npy", d)
                d = output_file[:-7] + ".npy"
        except Exception as e:
            print("Error in", l)
            print(e)
            errors_in.append(str(e))

    if errors_in:
        print("Errors occurred during processing:")
        for error in errors_in:
            print(error)

    return d,dct

# API route for prediction
@router.post("/predict/")
async def create_prediction(file: UploadFile):
    """Predict segmentation from the uploaded NIFTI file."""
    ModelDir = ModelDirT1


    filename = file.filename
    print("file name:-",filename)

    # Construct the full path where the file will be saved
    save_path = os.path.join(input_folder, filename)
    try:
        # Save the uploaded file with the same name as it was inputted
        with open(save_path, 'wb') as tmp:
            data = await file.read()
            tmp.write(data)

        # Prepare directories and model
        shutil.copy(join(ModelDir, 'plans.pkl'), output_folder)
        expected_num_modalities = load_pickle(join(ModelDir, "plans.pkl"))['num_modalities']
        case_ids = check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)
        
        # Prepare output files
        list_of_lists, cleaned_output_files = prepare_output_files(input_folder, case_ids, output_folder)
        
        # Clear CUDA cache
        print("Emptying CUDA cache")
        torch.cuda.empty_cache()

        # Load model
        trainer, params = load_model_and_checkpoint_files(ModelDir, folds, mixed_precision=mixed_precision, checkpoint_name=checkpoint_name)

        # Set export parameters
        force_separate_z, interpolation_order, interpolation_order_z = set_segmentation_export_params(trainer)
        
        # Process the prediction
        d, dct = process_prediction(trainer, list_of_lists, cleaned_output_files, params)

        print("Predicting", cleaned_output_files[0])
        trainer.load_checkpoint_ram(params[0], False)

        softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
            d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
            step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
            mixed_precision=mixed_precision)[1]
        
        for p in params[1:]:
            trainer.load_checkpoint_ram(p, False)
            softmax += trainer.predict_preprocessed_data_return_seg_and_softmax(
                d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
                mixed_precision=mixed_precision)[1]
        
        if len(params) > 1:
            softmax /= len(params)
        
        transpose_forward = trainer.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get('transpose_backward')
            softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])
        
        if hasattr(trainer, 'regions_class_order'):
            region_class_order = trainer.regions_class_order
        else:
            region_class_order = None
        
        np.save(cleaned_output_files[0][:-7] + ".npy", softmax)
        save_segmentation_nifti_from_softmax(softmax, cleaned_output_files[0], dct, interpolation_order, region_class_order,
                                    None, None,
                                    npz_file, None, force_separate_z, interpolation_order_z)
        
        if not disable_postprocessing:
            # results = []
            pp_file = join(ModelDir, "postprocessing.json")
            if isfile(pp_file):
                print("postprocessing...")
                shutil.copy(pp_file, os.path.abspath(os.path.dirname(cleaned_output_files[0])))
                # for_which_classes stores for which of the classes everything but the largest connected component needs to be
                # removed
                for_which_classes, min_valid_obj_size = load_postprocessing(pp_file)
                largest_removed, kept_size = load_remove_save(cleaned_output_files[0], cleaned_output_files[0], for_which_classes , min_valid_obj_size)
            else:
                print("WARNING! Cannot run postprocessing because the postprocessing file is missing. Make sure to run "
                    "consolidate_folds in the output folder of the model first!\nThe folder you need to run this in is "
                    "%s" % ModelDir)
        
        return FileResponse(cleaned_output_files[0], media_type="application/gzip", filename=os.path.basename(cleaned_output_files[0]))

        # return {"message": "success"}

    finally:
        pass
        # Clean up the temporary file
        # clean_up_temp_file(fd, niftyPath)









@router.post("/predictT2/")
async def create_prediction(file: UploadFile):
    """Predict segmentation from the uploaded NIFTI file."""
    ModelDir = ModelDirT2


    filename = file.filename
    print("file name:-",filename)

    # Construct the full path where the file will be saved
    save_path = os.path.join(input_folder, filename)
    try:
        # Save the uploaded file with the same name as it was inputted
        with open(save_path, 'wb') as tmp:
            data = await file.read()
            tmp.write(data)

        # Prepare directories and model
        shutil.copy(join(ModelDir, 'plans.pkl'), output_folder)
        expected_num_modalities = load_pickle(join(ModelDir, "plans.pkl"))['num_modalities']
        case_ids = check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)
        
        # Prepare output files
        list_of_lists, cleaned_output_files = prepare_output_files(input_folder, case_ids, output_folder)
        
        # Clear CUDA cache
        print("Emptying CUDA cache")
        torch.cuda.empty_cache()

        # Load model
        trainer, params = load_model_and_checkpoint_files(ModelDir, folds, mixed_precision=mixed_precision, checkpoint_name=checkpoint_name)

        # Set export parameters
        force_separate_z, interpolation_order, interpolation_order_z = set_segmentation_export_params(trainer)
        
        # Process the prediction
        d, dct = process_prediction(trainer, list_of_lists, cleaned_output_files, params)

        print("Predicting", cleaned_output_files[0])
        trainer.load_checkpoint_ram(params[0], False)

        softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
            d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
            step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
            mixed_precision=mixed_precision)[1]
        
        for p in params[1:]:
            trainer.load_checkpoint_ram(p, False)
            softmax += trainer.predict_preprocessed_data_return_seg_and_softmax(
                d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
                mixed_precision=mixed_precision)[1]
        
        if len(params) > 1:
            softmax /= len(params)
        
        transpose_forward = trainer.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get('transpose_backward')
            softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])
        
        if hasattr(trainer, 'regions_class_order'):
            region_class_order = trainer.regions_class_order
        else:
            region_class_order = None
        
        np.save(cleaned_output_files[0][:-7] + ".npy", softmax)
        save_segmentation_nifti_from_softmax(softmax, cleaned_output_files[0], dct, interpolation_order, region_class_order,
                                    None, None,
                                    npz_file, None, force_separate_z, interpolation_order_z)
        
        if not disable_postprocessing:
            # results = []
            pp_file = join(ModelDir, "postprocessing.json")
            if isfile(pp_file):
                print("postprocessing...")
                shutil.copy(pp_file, os.path.abspath(os.path.dirname(cleaned_output_files[0])))
                # for_which_classes stores for which of the classes everything but the largest connected component needs to be
                # removed
                for_which_classes, min_valid_obj_size = load_postprocessing(pp_file)
                largest_removed, kept_size = load_remove_save(cleaned_output_files[0], cleaned_output_files[0], for_which_classes , min_valid_obj_size)
            else:
                print("WARNING! Cannot run postprocessing because the postprocessing file is missing. Make sure to run "
                    "consolidate_folds in the output folder of the model first!\nThe folder you need to run this in is "
                    "%s" % ModelDir)
        
        return FileResponse(cleaned_output_files[0], media_type="application/gzip", filename=os.path.basename(cleaned_output_files[0]))

        # return {"message": "success"}

    finally:
        pass
        # Clean up the temporary file
        # clean_up_temp_file(fd, niftyPath)






























def set_segmentation_export_params(trainer) -> Tuple[Union[int, None], int, int]:
    """Set segmentation export parameters based on the trainer's plans."""
    segmentation_export_kwargs = None  # Define this as per requirement
    if segmentation_export_kwargs is None:
        if 'segmentation_export_params' in trainer.plans.keys():
            force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
            interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
            interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0
    else:
        force_separate_z = segmentation_export_kwargs['force_separate_z']
        interpolation_order = segmentation_export_kwargs['interpolation_order']
        interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']
    
    return force_separate_z, interpolation_order, interpolation_order_z
