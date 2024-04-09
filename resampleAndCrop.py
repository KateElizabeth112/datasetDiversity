# Resample and crop the images in a target folder
import nibabel as nib
from nibabel.processing import resample_to_output
import os
import argparse
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

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

    ds = "Dataset999_Full"

    organ_dict = {"background": 0,
                  "right kidney": 1,
                  "left kidney": 2,
                  "liver": 3,
                  "pancreas": 4}

elif dataset == "AMOS":
    if local == "local":
        root_dir = "/Users/katecevora/Documents/PhD/data/AMOS_3D"
    elif local == "remote":
        root_dir = "/rds/general/user/kc2322/home/data/AMOS_3D"
    else:
        raise Exception("Local flag is not recognised. Must be set to local or remote.")

    ds = "Dataset200_AMOS"

    organ_dict = {"background": 0,
                  "right kidney": 2,
                  "left kidney": 3,
                  "liver": 6,
                  "pancreas": 10}

image_dir = os.path.join(root_dir, "nnUNet_raw", ds, "imagesTr")
label_dir = os.path.join(root_dir, "nnUNet_raw", ds, "labelsTr")
target_spacing = (1.5, 1.5, 1.5)        # target voxel size

def resample():
    filenames = os.listdir(image_dir)

    for fn in filenames:
        if fn.endswith(".nii.gz"):
            # Load the original image
            original_img = nib.load(os.path.join(image_dir, fn))

            original_spacing = original_img.header.get_zooms()

            if np.max(np.abs(np.array(original_spacing) - np.array(target_spacing))) > 0.0001:
                print("Resampling of image {} required".format(fn))
                # Resample the image to the target voxel sizes
                #resampled_img = resample_to_output(original_img, voxel_sizes=target_voxel_sizes)

                # Save the resampled image
                #nib.save(resampled_img, 'resampled_image.nii.gz')

                # TODO remember to also resample the label


def getExtents(organ):
    # Collect the x, y and z extents of the organ with index idx so we can determine the crop window for the whole DS
    filenames = os.listdir(label_dir)

    organ_idx = organ_dict.get(organ)

    # create some containers to store the x, y and z extents
    x_extents = []
    y_extents = []
    z_extents = []

    for fn in filenames:
        if fn.endswith(".nii.gz"):
            # Load the original label
            label_nii = nib.load(os.path.join(label_dir, fn))

            # check that the spacing matches the target spacing
            spacing = label_nii.header.get_zooms()

            # convert label to numpy array
            label = label_nii.get_fdata()

            if np.max(np.abs(np.array(spacing) - np.array(target_spacing))) > 0.0001:
                raise Exception("The label {} voxel spacing of {} does not match the target {} ".format(fn,
                                                                                                        spacing,
                                                                                                        target_spacing))

            # Find the x, y and z extents of the organ with index idx
            # Filter out the label idx only
            label_idx = np.zeros(label.shape)
            label_idx[label == organ_idx] = 1

            # Check that this sums to more than zero
            if np.sum(label_idx) > 0:
                # sum along each pair of axes to find the maximum extent for the third axis
                x_sum = np.sum(label_idx, axis=1)
                y_sum = np.sum(label_idx, axis=2)
                z_sum = np.sum(label_idx, axis=0)

                x_extents.append(np.max(x_sum))
                y_extents.append(np.max(y_sum))
                z_extents.append(np.max(z_sum))
            else:
                print("The organ with index {} does not have any labels for image {}".format(organ_idx, fn))

    # save the x, y and z extents
    f = open(os.path.join(root_dir, "{}_extents.pkl".format(organ)), "wb")
    pkl.dump([x_extents, y_extents, z_extents], f)
    f.close()

    # print the maximum
    print("Maximum {0} extents (x, y, z) in voxels: ({1}, {2}, {3})".format(organ,
                                                                        np.max(np.array(x_extents)),
                                                                        np.max(np.array(y_extents)),
                                                                        np.max(np.array(z_extents))))


def crop(organ, crop_extent):
    # Crop the images to a window of size crop_extend, centering the organ in the volume
    print("Crop images")

    filenames = os.listdir(label_dir)

    organ_idx = organ_dict.get(organ)

    for fn in filenames:
        if fn.endswith(".nii.gz"):
            # Load the original label
            label_nii = nib.load(os.path.join(label_dir, fn))

            # convert label to numpy array
            label = label_nii.get_fdata()

            # Filter out the label idx only
            label_idx = np.zeros(label.shape)
            label_idx[label == organ_idx] = 1

            # Check that this sums to more than zero
            if np.sum(label_idx) > 0:
                # sum along each axis to find the maximum extent for the third axis
                x_sum = np.sum(label_idx, axis=(1, 2))
                y_sum = np.sum(label_idx, axis=(0, 2))
                z_sum = np.sum(label_idx, axis=(0, 1))

                # find the index of the start and end point, and the origin as the midpoint
                x_1 = np.nonzero(x_sum)[0][0]
                x_2 = np.nonzero(x_sum)[0][-1]
                x_0 = int(np.round(0.5 * (x_1 + x_2)))

                y_1 = np.nonzero(y_sum)[0][0]
                y_2 = np.nonzero(y_sum)[0][-1]
                y_0 = int(np.round(0.5 * (y_1 + y_2)))

                z_1 = np.nonzero(z_sum)[0][0]
                z_2 = np.nonzero(z_sum)[0][-1]
                z_0 = int(np.round(0.5 * (z_1 + z_2)))

                # get the bounding box for the x_z_plane
                x_min = int(x_0 - crop_extent[0] / 2)
                x_max = int(x_0 + crop_extent[0] / 2)

                z_min = int(z_0 - crop_extent[2] / 2)
                z_max = int(z_0 + crop_extent[2] / 2)

                # plot the x_z plane sliced through the y-origin
                x_z_plane = label_idx[:, y_0, :]
                plt.clf()
                plt.imshow(np.rot90(x_z_plane), origin='lower')
                #plt.axvline(z_min)
                #plt.axvline(z_max)
                #plt.axhline(x_min)
                #plt.axhline(x_max)
                #plt.scatter(z_0, x_0)
                plt.axvline(x_0)
                #plt.axvline(x_1)
                #plt.axvline(x_2)
                plt.show()



def main():
    #resample()
    getExtents("liver")
    #crop("liver", (60, 60, 60))

if __name__ == "__main__":
    main()