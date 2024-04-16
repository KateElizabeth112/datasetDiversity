# Resample and crop the images in a target folder
import nibabel as nib
import os
import argparse
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from gzip import BadGzipFile
import SimpleITK as sitk

parser = argparse.ArgumentParser(description="Just an example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dataset", default="AMOS", help="Task to evaluate")
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

    target_spacing = [1.5, 1.5, 1.5]  # target voxel size

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

    target_spacing = [0.7, 0.7, 2.0]  # target voxel size

image_dir = os.path.join(root_dir, "nnUNet_raw", ds, "imagesTr")
label_dir = os.path.join(root_dir, "nnUNet_raw", ds, "labelsTr")

image_dir_resampled = os.path.join(root_dir, "nnUNet_raw", ds, "imagesTr_resampled")
label_dir_resampled = os.path.join(root_dir, "nnUNet_raw", ds, "labelsTr_resampled")


def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def resample():
    filenames = os.listdir(image_dir)

    for fn in filenames:
        if fn.endswith(".nii.gz"):
            # Load the original image
            image_sitk = sitk.ReadImage(os.path.join(image_dir, fn))
            original_spacing = image_sitk.GetSpacing()

            # Check whether we need to resample (if the original and target spacing differ)
            if np.max(np.abs(np.array(original_spacing) - np.array(target_spacing))) > 0.0001:
                print("Resampling of image {} required".format(fn))

                # Resample the image to the target voxel sizes
                image_resampled = resample_img(image_sitk, out_spacing=target_spacing, is_label=False)

                # load the label and resample
                label_sitk = sitk.ReadImage(os.path.join(label_dir, fn[0:9] + ".nii.gz"))
                label_resampled = resample_img(label_sitk, out_spacing=target_spacing, is_label=True)

                # Make some checks
                assert np.max(np.abs(np.array(image_resampled.GetSpacing()) - np.array(
                    target_spacing))) < 0.0001, "Spacing of resampled image voxels does not match target"

                assert np.max(np.abs(np.array(label_resampled.GetSpacing()) - np.array(
                    target_spacing))) < 0.0001, "Spacing of resampled label voxels does not match target"

                assert label_resampled.GetSize() == image_resampled.GetSize(), "Resampled image and label are not the same size"

                # debug only, check how the images look before and after resampling
                organs = ["right kidney", "left kidney", "liver", "pancreas"]
                for o in organs:
                    organ_idx = organ_dict.get(o)

                    # convert label and image to numpy array
                    image_np = sitk.GetArrayFromImage(image_sitk)
                    label_np = sitk.GetArrayFromImage(label_sitk)

                    # filter the label for just one organ
                    label_np_o = np.zeros(label_np.shape)
                    label_np_o[label_np == organ_idx] = 1

                    # plot each plane sliced through the centre of the image
                    plt.clf()
                    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 6))
                    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

                    ax1.imshow(np.rot90(label_np_o[int(label_np_o.shape[0] / 2), :, :]), origin='lower')
                    ax2.imshow(np.rot90(label_np_o[:, int(label_np_o.shape[1] / 2), :]), origin='lower')
                    ax3.imshow(np.rot90(label_np_o[:, :, int(label_np_o.shape[2] / 2)]), origin='lower')

                    ax4.imshow(np.rot90(image_np[int(image_np.shape[0] / 2), :, :]), origin='lower', cmap="gray")
                    ax5.imshow(np.rot90(image_np[:, int(image_np.shape[1] / 2), :]), origin='lower', cmap="gray")
                    ax6.imshow(np.rot90(image_np[:, :, int(image_np.shape[2] / 2)]), origin='lower', cmap="gray")
                    plt.suptitle(o)

                    # save the image
                    # check we have a directory to save the images
                    if not os.path.exists(os.path.join(root_dir, "images", "resampled")):
                        os.mkdir(os.path.join(root_dir, "images", "resampled"))

                    if not os.path.exists(os.path.join(root_dir, "images", "resampled", o)):
                        os.mkdir(os.path.join(root_dir, "images", "resampled", o))

                    plt.savefig(os.path.join(root_dir, "images", "resampled", o, fn[:9] + ".png"))
                    plt.close()

                    # convert label and image to numpy array
                    image_np_resampled = sitk.GetArrayFromImage(image_resampled)
                    label_np_resampled = sitk.GetArrayFromImage(label_resampled)

                    # filter the label for just one organ
                    label_np_o_resampled = np.zeros(label_np_resampled.shape)
                    label_np_o_resampled[label_np_resampled == organ_idx] = 1

                    # plot each plane sliced through the centre of the image
                    plt.clf()
                    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 6))
                    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

                    ax1.imshow(np.rot90(label_np_o_resampled[int(label_np_o_resampled.shape[0] / 2), :, :]), origin='lower')
                    ax2.imshow(np.rot90(label_np_o_resampled[:, int(label_np_o_resampled.shape[1] / 2), :]), origin='lower')
                    ax3.imshow(np.rot90(label_np_o_resampled[:, :, int(label_np_o_resampled.shape[2] / 2)]), origin='lower')

                    ax4.imshow(np.rot90(image_np_resampled[int(image_np_resampled.shape[0] / 2), :, :]), origin='lower', cmap="gray")
                    ax5.imshow(np.rot90(image_np_resampled[:, int(image_np_resampled.shape[1] / 2), :]), origin='lower', cmap="gray")
                    ax6.imshow(np.rot90(image_np_resampled[:, :, int(image_np_resampled.shape[2] / 2)]), origin='lower', cmap="gray")
                    plt.suptitle("{} Resampled".format(o))

                    plt.savefig(os.path.join(root_dir, "images", "resampled", o, fn[:9] + "_resampled.png"))
                    plt.close()

                # Save the resampled image and label in a new folder
                # check we have a directory to save the images
                if not os.path.exists(os.path.join(root_dir, "nnUNet_raw", ds, "imagesTr_resampled")):
                    os.mkdir(os.path.join(root_dir, "nnUNet_raw", ds, "imagesTr_resampled"))

                if not os.path.exists(os.path.join(root_dir, "nnUNet_raw", ds, "labelsTr_resampled")):
                    os.mkdir(os.path.join(root_dir, "nnUNet_raw", ds, "labelsTr_resampled"))

                sitk.WriteImage(image_resampled, os.path.join(root_dir, "nnUNet_raw", ds, "imagesTr_resampled", fn))
                sitk.WriteImage(label_resampled, os.path.join(root_dir, "nnUNet_raw", ds, "labelsTr_resampled", fn))


def getExtents(organ):
    # Collect the x, y and z extents of the organ with index idx so we can determine the crop window for the whole DS
    if dataset == "TS":
        filenames = os.listdir(label_dir)
    else:
        filenames = os.listdir(label_dir_resampled)

    organ_idx = organ_dict.get(organ)

    # create some containers to store the x, y and z extents
    x_extents = []
    y_extents = []
    z_extents = []

    for fn in filenames:
        if fn.endswith(".nii.gz"):
            try:
                # Load the original label
                if dataset == "TS":
                    label_nii = nib.load(os.path.join(label_dir, fn))
                else:
                    label_nii = nib.load(os.path.join(label_dir_resampled, fn))
            except BadGzipFile:
                print("The file {} did not pass CRC check".format(fn))
            except Exception as e:
                print("An unexpected error {} occured".format(e))
            else:
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

    # Choose the maximum extents
    #crop_extent = [int(np.max(np.array(x_extents))), int(np.max(np.array(y_extents))), int(np.max(np.array(z_extents)))]

    # Choose 95th percentile extent
    per = 95
    crop_extent = [int(np.percentile(np.array(x_extents), per)),
                   int(np.percentile(np.array(y_extents), per)),
                   int(np.percentile(np.array(z_extents), per))]

    # Make sure they are divisible by 2
    for j in range(0, len(crop_extent)):
        if (crop_extent[j] % 2) > 0:
            crop_extent[j] = crop_extent[j] + 1

    # print the maximum extents
    print("Maximum {0} extents (x, y, z) in voxels: ({1}, {2}, {3})".format(organ,
                                                                            crop_extent[0],
                                                                            crop_extent[1],
                                                                            crop_extent[2]))

    # Save them somewhere for reading back
    f = open(os.path.join(root_dir, "{}_crop_extent.pkl".format(organ)), "wb")
    pkl.dump(crop_extent, f)
    f.close()


def crop(organ):
    # Crop the images to a window of size crop_extend, centering the organ in the volume
    print("Crop images")

    if dataset == "TS":
        filenames = os.listdir(label_dir)
    else:
        filenames = os.listdir(label_dir_resampled)
    organ_idx = organ_dict.get(organ)

    # open the crop extents for the dataset and the organ
    f = open(os.path.join(root_dir, "{}_crop_extent.pkl".format(organ)), "rb")
    crop_extent = pkl.load(f)
    f.close()

    for fn in filenames:
        if fn.endswith(".nii.gz"):
            # Load the original label
            try:
                if dataset == "TS":
                    label_nii = nib.load(os.path.join(label_dir, fn))
                    # load the corresponding image
                    image_nii = nib.load(os.path.join(image_dir, fn[:9] + "_0000.nii.gz"))
                else:
                    label_nii = nib.load(os.path.join(label_dir_resampled, fn))
                    # load the corresponding image
                    image_nii = nib.load(os.path.join(image_dir_resampled, fn[:9] + "_0000.nii.gz"))
                label = label_nii.get_fdata()  # convert label to numpy array
                image = image_nii.get_fdata()  # convert image to numpy array
            except BadGzipFile:
                print("File {} failed CRC check".format(fn))
            except Exception as e:
                print("An unexpected error {} occured".format(e))
            else:
                # Filter out the label idx only
                label_idx = np.zeros(label.shape)
                label_idx[label == organ_idx] = 1

                # Check that this sums to more than zero
                if np.sum(label_idx) > 0:
                    # sum along each pair of axes to find the plane with maximum area for the third axis
                    x_area = np.sum(label_idx, axis=(1, 2))
                    y_area = np.sum(label_idx, axis=(0, 2))
                    z_area = np.sum(label_idx, axis=(0, 1))

                    # find the index of the start and end point, and the origin as the midpoint
                    x_1 = np.nonzero(x_area)[0][0]
                    x_2 = np.nonzero(x_area)[0][-1]
                    x_0 = int(np.round(0.5 * (x_1 + x_2)))

                    y_1 = np.nonzero(y_area)[0][0]
                    y_2 = np.nonzero(y_area)[0][-1]
                    y_0 = int(np.round(0.5 * (y_1 + y_2)))

                    z_1 = np.nonzero(z_area)[0][0]
                    z_2 = np.nonzero(z_area)[0][-1]
                    z_0 = int(np.round(0.5 * (z_1 + z_2)))

                    # get the bounding box for all planes
                    x_min = int(x_0 - crop_extent[0] / 2)
                    x_max = int(x_0 + crop_extent[0] / 2)

                    y_min = int(y_0 - crop_extent[1] / 2)
                    y_max = int(y_0 + crop_extent[1] / 2)

                    z_min = int(z_0 - crop_extent[2] / 2)
                    z_max = int(z_0 + crop_extent[2] / 2)

                    # crop the label and image
                    label_crop = label_idx[np.max((0, x_min)):x_max, np.max((0, y_min)):y_max, np.max((0, z_min)):z_max]
                    image_crop = image[np.max((0, x_min)):x_max, np.max((0, y_min)):y_max, np.max((0, z_min)):z_max]

                    # check if we need to do any padding (i.e. if the cropping planes are outside bounds of original image)
                    if np.sum(crop_extent) > np.sum(label_crop.shape):

                        if x_min >= 0:
                            x_min_offset = 0
                        elif x_min < 0:
                            x_min_offset = abs(x_min)

                        if y_min >= 0:
                            y_min_offset = 0
                        elif y_min < 0:
                            y_min_offset = abs(y_min)

                        if z_min >= 0:
                            z_min_offset = 0
                        elif z_min < 0:
                            z_min_offset = abs(z_min)

                        label_crop_padded = np.zeros(crop_extent)
                        image_crop_padded = np.zeros(crop_extent)

                        label_crop_padded.fill(0.5)
                        image_crop_padded.fill(np.mean(image_crop))

                        label_crop_padded[x_min_offset:x_min_offset + label_crop.shape[0],
                        y_min_offset:y_min_offset + label_crop.shape[1],
                        z_min_offset:z_min_offset + label_crop.shape[2]] = label_crop

                        image_crop_padded[x_min_offset:x_min_offset + label_crop.shape[0],
                        y_min_offset:y_min_offset + label_crop.shape[1],
                        z_min_offset:z_min_offset + label_crop.shape[2]] = image_crop

                        label_crop = label_crop_padded
                        image_crop = image_crop_padded

                    # plot each plane sliced through the centre of the image
                    plt.clf()
                    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 6))
                    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

                    ax1.imshow(np.rot90(label_crop[int(crop_extent[0] / 2), :, :]), origin='lower')
                    ax2.imshow(np.rot90(label_crop[:, int(crop_extent[1] / 2), :]), origin='lower')
                    ax3.imshow(np.rot90(label_crop[:, :, int(crop_extent[2] / 2)]), origin='lower')

                    ax4.imshow(np.rot90(image_crop[int(crop_extent[0] / 2), :, :]), origin='lower', cmap="gray")
                    ax5.imshow(np.rot90(image_crop[:, int(crop_extent[1] / 2), :]), origin='lower', cmap="gray")
                    ax6.imshow(np.rot90(image_crop[:, :, int(crop_extent[2] / 2)]), origin='lower', cmap="gray")

                    # save the image
                    # check we have a directory to save the images
                    if not os.path.exists(os.path.join(root_dir, "images", "crops")):
                        os.mkdir(os.path.join(root_dir, "images", "crops"))

                    if not os.path.exists(os.path.join(root_dir, "images", "crops", organ)):
                        os.mkdir(os.path.join(root_dir, "images", "crops", organ))

                    plt.savefig(os.path.join(root_dir, "images", "crops", organ, fn[:9] + ".png"))
                    plt.close()


def main():
    resample()
    organs = ["left kidney", "right kidney", "liver", "pancreas"]

    for organ in organs:
        getExtents(organ)
        #crop(organ)


if __name__ == "__main__":
    main()
