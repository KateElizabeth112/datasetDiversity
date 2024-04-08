# Resample and crop the images in a target folder
import nibabel as nib
from nibabel.processing import resample_to_output
import os
import argparse

parser = argparse.ArgumentParser(description="Just an example",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dataset", default="TS", help="Task to evaluate")
parser.add_argument("-l", "--local", default="local", help="Are we running on local or remote")
args = vars(parser.parse_args())

dataset = args["dataset"]
local = args["local"]

if dataset == "TS":
    if local == "local":
        root_dir = "/Users/katecevora/Documents/PhD/data/TotalSegmentator"
    elif local == "remote":
        root_dir = "/rds/general/user/kc2322/projects/cevora_phd/live/TotalSegmentator"
    else:
        raise Exception("Local flag is not recognised. Must be set to local or remote.")

    image_dir = os.path.join(root_dir, "nnUNet_raw", "Dataset999_Full", "imagesTr")
    gt_seg_dir = os.path.join(root_dir, "nnUNet_raw", "Dataset999_Full", "labelsTr")

elif dataset == "AMOS":
    if local == "local":
        root_dir = "/Users/katecevora/Documents/PhD/data/AMOS_3D"
    elif local == "remote":
        root_dir = "/rds/general/user/kc2322/home/data/AMOS_3D"
    else:
        raise Exception("Local flag is not recognised. Must be set to local or remote.")
    image_dir = os.path.join(root_dir, "nnUNet_raw", "Dataset200_AMOS", "imagesTr")
    gt_seg_dir = os.path.join(root_dir, "nnUNet_raw", "Dataset200_AMOS", "labelsTr")


def resample():
    filenames = os.listdir(image_dir)

    for fn in filenames:
        if fn.endswith(".nii.gz"):

            # Load the original image
            original_img = nib.load(os.path.join(root_dir, "case_1179_0000.nii.gz"))

            original_vox_size = original_img.header.get_zooms()

            # Define the target voxel sizes
            target_voxel_sizes = (1.5, 1.5, 1.5)  # Define your desired voxel sizes here

            if original_vox_size != target_voxel_sizes:
                print("Resampling of image {} required".format(fn))
                # Resample the image to the target voxel sizes
                #resampled_img = resample_to_output(original_img, voxel_sizes=target_voxel_sizes)

                # Save the resampled image
                #nib.save(resampled_img, 'resampled_image.nii.gz')