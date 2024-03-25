import plotly.graph_objs as go
import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import os

organ_dict = {1: "left kidney", 2: "right kidney", 3: "liver", 4: "pancreas"}


# Plot a 3D mesh from a binary  3D label alongside the ground truth
def plot3Dmesh(gt, volumes, save_path, subject):
    fig = go.Figure()

    # lighting settings for PlotLy objects
    lighting = dict(ambient=0.5, diffuse=0.5, roughness=0.5, specular=0.6, fresnel=0.8)

    # cycle over the organs and plot layers
    values = np.unique(gt)
    for k in list(values):
        if k > 0:
            gt_k = np.zeros(gt.shape)
            gt_k[gt == k] = 1

            try:
                gt_verts, gt_faces, gt_normals, gt_values = measure.marching_cubes(gt_k, 0)
                gt_x, gt_y, gt_z = gt_verts.T
                gt_I, gt_J, gt_K = gt_faces.T
                gt_mesh = go.Mesh3d(x=gt_x, y=gt_y, z=gt_z,
                                    intensity=gt_values,
                                    i=gt_I, j=gt_J, k=gt_K,
                                    lighting=lighting,
                                    name=organ_dict[k],
                                    showscale=False,
                                    opacity=1.0,
                                    colorscale='magma'
                                    )
                fig.add_trace(gt_mesh)
            except:
                print("GT mesh extraction failed")

            fig.update_layout(title_text="{0} volume: {1:.0f} ml".format(organ_dict[k], volumes[int(k)-1] / 1000))
            fig.update_xaxes(visible=False, showticklabels=False)
            fig.write_image(os.path.join(save_path, "{}_{}.png".format(organ_dict[k], subject)))


def plotSlices(image, label, save_path):
    # plot slices in 3-axes

    # get centre voxel
    s = np.array(image.shape)
    c = np.around(s / 2).astype(int)

    rot_k = 1
    # take slices of the image and the label
    image_x = np.rot90(image[c[0], ::-1, :], k=rot_k)
    label_x = np.rot90(label[c[0], ::-1, :], k=rot_k)

    image_y = np.rot90(image[::-1, c[1], :], k=rot_k)
    label_y = np.rot90(label[::-1, c[1], :], k=rot_k)

    image_z = np.rot90(image[::-1, ::-1, c[2]], k=rot_k)
    label_z = np.rot90(label[::-1, ::-1, c[2]], k=rot_k)

    # create masks so that the label is transparent for the background
    mask_x = np.zeros(label_x.shape)
    mask_x[label_x > 0] = 0.5

    mask_y = np.zeros(label_y.shape)
    mask_y[label_y > 0] = 0.5

    mask_z = np.zeros(label_z.shape)
    mask_z[label_z > 0] = 0.5

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 2))
    ax1, ax2, ax3 = axes.flatten()

    # put everything into a list so we can iterate over them when creating subplots
    axes = [ax1, ax2, ax3]
    images = [image_x, image_y, image_z]
    labels = [label_x, label_y, label_z]
    masks = [mask_x, mask_y, mask_z]
    titles = ["Sagittal (X-plane)", "Coronal (Y-plane)", "Axial (Z-plane)"]

    for i in range(3):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].imshow(labels[i], cmap="jet", alpha=masks[i], vmin=0, vmax=4)
        axes[i].axis('off')
        axes[i].set_title(titles[i])

    plt.tight_layout()
    plt.savefig(save_path)
