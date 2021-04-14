import os
import pickle
from scipy.spatial.transform import Rotation as R
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import k3d
import seaborn as sns

DATA_ROOT_PATH = 'data/'
MODEL_SAVES_PATH = DATA_ROOT_PATH + "saved_models/"

def check_create_folder(folder_path):
    if not os.path.exists(os.path.dirname(folder_path)):
        try:
            os.makedirs(os.path.dirname(folder_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def create_metadata(path, **kwargs):
    """
    Creates the metadata file at path/metadata/metadata.pkl

    Args:
        path: path to metadata folder
        kwargs: each kwarg is stored in the dict saved as pickle file

    """
    meta_path = f'{path}metadata/'
    check_create_folder(meta_path)
    with open(f'{meta_path}metadata.pickle', 'wb') as f:
        d = {x[0]: x[1] for x in kwargs.items()}
        pickle.dump(d, f)
    print(f'Metadata Stored : {d}')


def read_metadata(path):
    """
    Read the metadata file at path/metadata/metadata.pkl

    Args:
        path: path to metadata folder

    Returns:
        dict contained in metadata file

    """
    meta_file = f'{path}metadata/metadata.pickle'
    with open(meta_file, 'rb') as f:
        return pickle.load(f)

def rotate(points, angles):
    """
        Rotates a set of point around 'xyz' for the angles given in radians.

    Args:
        points: array of 3d points, with shape (n_points, 3)
        angles: angles[0]: rotation around x
                angles[1]: rotation around y
                angles[2]: rotation around z

    Returns:
        rotated points

    """
    r = R.from_euler('xyz', angles, degrees=False)
    return r.apply(points)

def separated_multi_auc(pred, label, num_labels):
    """
        Computes the AUC

    Args:
        pred: torch tensor of predictions with dimension [n_samples, n_labels] each prediction
            has to be a probability
        label: torch tensor of dimension `[n_samples]` of ground truth
            labels NOT one-hot encoded

    Returns:
        AUC of the prediction computed with sklearn roc_auc_score with parameters `multi_class="ovr"` and `averege="macro"`
    """
    np_pred = pred.cpu().detach().numpy()
    np_label = label.cpu().numpy().astype(np.int64)
    # num_labels = len(np.unique(np_label))
    np_label_one_hot = np.zeros((np_label.size, num_labels))
    # print(num_labels)
    np_label_one_hot[np.arange(np_label.size), np_label] = 1
    ret = {}
    for label_ind in range(num_labels):
        ret[label_ind] = roc_auc_score(y_score=np_pred[:, label_ind],
                                        y_true=np_label_one_hot[:, label_ind])
    return ret


name_mapping = {
    0: "terrain",
    1: "construction",
    2: "urban_asset",
    3: "vegetation",
    4: "vehicle",
}
palette = sns.color_palette("pastel")
# map at least green to vegetation
g = palette[2]
palette[2] = palette[4]
palette[4] = g
# create color mapping in a format which is suitable for k3d
color_mapping = {
    l: [int(palette[i][0] * 255),
        int(palette[i][1] * 255),
        int(palette[i][2] * 255)]
    for i, l in enumerate(name_mapping)
}

def pack(r, g, b):
    """
        (r,g,b) tuple to hex. Each r,g,b can be column arrays

    """
    return (
            (np.array(r).astype(np.uint32) << 16)
            + (np.array(g).astype(np.uint32) << 8)
            + np.array(b).astype(np.uint32)
    )


def pack_single(rgb):
    """
        [r,g,b] one line array to hex
    """
    return (
            (rgb[0] << 16)
            + (rgb[1] << 8)
            + rgb[2]
    )


def generate_k3d_plot(xyz, rgb=None, mask_map=None, mask_color=None,
                      name_map=None, old_plot=None):
    """
        Generates a k3d snapshot of a set of 3d points, mapping them either
        to their true rgb color or a colour corresponding to their label.
        Labels are also mapped to names, so that they can be easily toggled
        inside the visualization tool.

    Args:
        xyz: array of [x, y, z] points
        rgb: array of [r, g, b] points inside [0, 255]
        mask_map: dict mapping each label to a mask over xyz, which allows to
            select points from each class
        mask_color: dict mapping each label to a single color
        name_map: map each numeric label to a descriptive string name

    Returns:
        k3d snapshot that can be saved as html for visualization

    """
    kwargs = dict()
    if rgb is None:
        pass
    else:
        assert mask_color is None
        kwargs["colors"] = pack(rgb[:, 0], rgb[:, 1], rgb[:, 2])

    if old_plot is None:
        plot = k3d.plot()
    else:
        plot = old_plot

    if mask_map is None:
        plt_points = k3d.points(positions=xyz,
                                point_size=1.,
                                shader="flat",
                                **kwargs)
        plot += plt_points
    else:
        for label in mask_map:
            mask = mask_map[label]
            if name_map is None:
                legend_label = f"label {label}"
            else:
                legend_label = f"{name_map[label]}"
            if mask_color is None:
                colors = kwargs["colors"][mask]
                plt_points = k3d.points(positions=xyz[mask],
                                        point_size=1.,
                                        shader="flat",
                                        name=legend_label,
                                        colors=colors)
                plot += plt_points
            else:
                color = pack_single(mask_color[label])
                plt_points = k3d.points(
                    positions=xyz[mask],
                    point_size=1.,
                    shader="flat",
                    name=legend_label,
                    color=color,
                )
                plot += plt_points
    plot.camera_mode = 'orbit'
    plot.grid_auto_fit = False
    # plot.grid = np.concatenate((np.min(xyz, axis=0), np.max(xyz, axis=0)))
    plot.grid_visible = False
    return plot