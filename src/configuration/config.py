# ModelDir = "/home/omen/Documents/VS/resources/6827679/Single-Centre-Gamma-Knife-(SC-GK)-models/ceT1/nnUNet/3d_fullres/Task700_VS/nnUNetTrainerV2__nnUNetPlansv2.1"
ModelDirT1 = '/home/omen/Documents/VS/resources/6827679/Multi-Centre-Routine-Clinical-(MC-RC)-models/ceT1/nnUNet/3d_fullres/Task900_VS/nnUNetTrainerV2__nnUNetPlansv2.1'


ModelDirT2 = '/home/omen/Documents/VS/resources/6827679/Multi-Centre-Routine-Clinical-(MC-RC)-models/T2/nnUNet/3d_fullres/Task901_VS/nnUNetTrainerV2__nnUNetPlansv2.1'
input_folder = "InputFolder"
output_folder = "output_folder"
mixed_precision=True
# mixed_precision = False

checkpoint_name: str = "model_final_checkpoint"
folds = [0,1,2,3,4]
segmentation_export_kwargs = None
do_tta = True
step_size=0.5
all_in_gpu = True

bytes_per_voxel = 4
if all_in_gpu:
    bytes_per_voxel = 2  # if all_in_gpu then the return value is half (float16)

npz_file = None
disable_postprocessing = False
segs_from_prev_stage: dict = None