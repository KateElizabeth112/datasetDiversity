import os
import nibabel as nib
import numpy as np
import pickle as pkl
import argparse
from plottingFunctions import plotSlices

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

output_dir = os.path.join(root_dir, "images", "slices")

def main():
    # get a list of the files in the images folder
    f_names = os.listdir(image_dir)

    # create containers to store the volumes
    volumes_all = []
    subjects = []

    for f in f_names:
        if f.endswith(".nii.gz"):
            # load image
            subjects.append(f[5:9])
            image_nii = nib.load(os.path.join(image_dir, f))

            # load label
            label_nii = nib.load(os.path.join(gt_seg_dir, f[:9] + ".nii.gz"))

            # get the volume of 1 voxel in mm3
            sx, sy, sz = image_nii.header.get_zooms()
            vox_volume = sx * sy * sz

            # convert the image to a numpy array
            image = image_nii.get_fdata()
            label = label_nii.get_fdata()

            # if the label is from the AMOS dataset we will need to a conversion of the labels to match with TS
            if dataset == "AMOS":
                input_map = [2, 3, 6, 10]
                output_map = [1, 2, 3, 4]

                for q in range(len(input_map)):
                    label[label == input_map[q]] = output_map[q]

                label[label > 4] == 0

            # pass the image to the slice plotting function
            save_path = os.path.join(output_dir, "slices_{}.png".format(f[5:9]))
            plotSlices(image, label, save_path)


if __name__ == "__main__":
    main()





