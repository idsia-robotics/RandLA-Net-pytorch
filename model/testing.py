import pickle
import math
import glob

import numpy as np
import datetime

import torch.nn.functional as F
from torch import nn
from torch.utils import data
from tqdm import tqdm
import torch

from .dataset import RandlanetDataset
from .sampler import RandlanetWeightedSampler
from .model import RandlaNet
from .utils import create_metadata, read_metadata, check_create_folder, generate_k3d_plot, color_mapping, name_mapping
from .training import unpack_input


def segment(test_loader, model, device, inv_mapping, cfg, max_epoch=150):
    """
    Args:
        test_loader: todo
        model: pytorch loaded model
        device: pytorch device
        inv_mapping: dict to map net outputs to original labels
    """
    test_logger = tqdm(test_loader,
                       desc="Segmentation",
                       total=len(test_loader))
    n_points = len(test_loader.dataset)
    n_classes = len(inv_mapping)
    xyz_probs = np.zeros((n_points, n_classes))
    xyz_probs[:] = np.nan
    visited = np.zeros((n_points,), dtype=np.int32)
    model.eval()
    #test_smooth = 0.98
    n_votes = 2
    with torch.no_grad():
        for step in range(max_epoch):
            test_logger = tqdm(test_loader,
                               desc="Segmentation",
                               total=len(test_loader))
            print(f"Round {step}")
            for input_list in test_logger:
                inputs = unpack_input(input_list, cfg['num_layers'], device)
                outputs = model(inputs)
                outputs = F.log_softmax(outputs, dim=1)
                outputs = torch.reshape(outputs, [cfg['val_batch_size'], -1, cfg['num_classes']])

                for j in range(outputs.shape[0]):
                    probs = outputs[j, :, :].cpu().detach().float().numpy()
                    # probs = np.swapaxes(np.squeeze(probs), 0, 1)
                    ids = inputs['input_inds'][j, :].cpu().detach().int().numpy()
                    xyz_probs[ids] = np.nanmean([xyz_probs[ids], np.exp(probs)], axis=0)
                    # xyz_probs[ids] = test_smooth * xyz_probs[ids] \
                    #     + (1 - test_smooth) * probs
                    visited[ids] += 1
            least_visited = np.min(np.unique(visited))
            if least_visited >= n_votes:
                print(f"Each point was visited at least {n_votes}")
                break
            else:
                print(least_visited)
    for pc_id in test_loader.dataset.kdtrees:
        xyz_tile = test_loader.dataset.kdtrees[pc_id].data
        true_rgb = test_loader.dataset.colors[pc_id]*255.
        gt_labels = test_loader.dataset.labels[pc_id]
    xyz_labels = np.argmax(xyz_probs, axis=1)

    return xyz_tile, xyz_labels, xyz_probs, true_rgb, gt_labels


def store_results(model_path, xyz_tile, xyz_labels, xyz_probs, true_rgb,
                  gt_labels, pc_path, segmentation_name):
    """
        Stores segmentation results that will be used to generate analysis
        and to upload data to ISIN db

    Args:
        model_path: path to the model used for segmentation
        xyz_tile: [x,y,z] array of points
        xyz_labels: array of labels for each point
        xyz_probs: array of model outputs for each point
        true_rgb: [r,g,b] array for each point
        gt_labels: ground truth labels for each point
        pc_path: path containing segmented pc file
            segmented pc
        segmentation_name: name for the segmentation folder. If None, timestamp
            will be used

    """
    print("Storing segmentation results")
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if segmentation_name is None:
        segmentation_name = date

    results_path = f"{model_path}output/segmentations/{segmentation_name}/"
    check_create_folder(results_path)

    with open(f"{results_path}/xyz_tile.pickle", "wb") as pickle_out:
        pickle.dump(xyz_tile, pickle_out)

    with open(f"{results_path}/xyz_probs.pickle", "wb") as pickle_out:
        pickle.dump(xyz_probs, pickle_out)

    with open(f"{results_path}/xyz_labels.pickle", "wb") as pickle_out:
        pickle.dump(xyz_labels, pickle_out)

    with open(f"{results_path}/true_rgb.pickle", "wb") as pickle_out:
        pickle.dump(true_rgb, pickle_out)

    with open(f"{results_path}/gt_labels.pickle", "wb") as pickle_out:
        pickle.dump(gt_labels, pickle_out)

    metadata = read_metadata(pc_path)
    metadata['timestamp'] = date
    create_metadata(results_path, **metadata)
    print(f"Results stored at: {results_path}")
    return results_path


def segment_randlanet(model_path, pc_path, cfg, num_workers, segmentation_name=None):
    """Classify all the points contained in the provided pc using the best
    checkpoint of the selected model. It stores the results inside the
    model folder.

    Args:
        model_path: path to the folder containing all data generated during
            model training for a given model
        pc_path: path to the folder containing the pc
        segmentation_name: name for the segmentation folder. If None, timestamp
            will be used
    """
    with open(f"{model_path}output/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    mapping = metadata["label_mapping"]
    best_epoch = metadata["best_epoch"]
    model_name = glob.glob(f"{model_path}checkpoints/{best_epoch}_*.pth")[0]
    print(f"Loading model checkpoint: {model_name}")
    inv_mapping = {mapping[l]: l for l in mapping}
    print(f"Label inverse mapping: {inv_mapping}")

    n_classes = len(mapping)
    print(f"Best epoch was {metadata['best_epoch']}")
    print("Setting up pytorch")
    use_cuda = torch.cuda.is_available()
    print(f"Use cuda: {use_cuda}")
    device = torch.device("cuda:0" if use_cuda else "cpu")

    test_params = {"batch_size": cfg['val_batch_size'],
                   "shuffle": False,
                   "num_workers": num_workers}

    test_set = RandlanetDataset([pc_path], **cfg)
    test_sampler = RandlanetWeightedSampler(test_set,
                                            cfg['val_batch_size'] * cfg[
                                                'val_steps'])

    test_loader = data.DataLoader(test_set, sampler=test_sampler, **test_params)
    nice_model = model_name
    model = RandlaNet(n_layers=cfg['num_layers'], n_classes=cfg['num_classes'], d_out=cfg['d_out'])

    if not use_cuda:
        map_location = torch.device("cpu")
        model.load_state_dict(torch.load(nice_model, map_location=map_location))
    else:
        model.load_state_dict(torch.load(nice_model))
    model = model.to(device)

    xyz_tile, xyz_labels, xyz_probs, true_rgb, gt_labels = \
        segment(test_loader, model, device, inv_mapping, cfg)

    results_path = store_results(model_path, xyz_tile, xyz_labels, xyz_probs, true_rgb,
                   gt_labels, pc_path, segmentation_name)
    
    mask_map = {}
    for label in mapping.values():
        mask = xyz_labels == label
        mask_map[label] = mask

    plot = generate_k3d_plot(xyz_tile, mask_map=mask_map, mask_color=color_mapping, name_map=name_mapping)
    snapshot = plot.get_snapshot(9)
    snap_path = f"{results_path}snapshot_predictions.html"
    with open(snap_path, 'w') as fp:
        fp.write(snapshot)
        print(f"Labelled snapshot save at {snap_path}")

    plot = generate_k3d_plot(xyz_tile, rgb=true_rgb, mask_map=mask_map, name_map=name_mapping)
    snapshot = plot.get_snapshot(9)
    snap_path = f"{results_path}snapshot_rgb.html"
    with open(snap_path, 'w') as fp:
        fp.write(snapshot)
        print(f"RGB snapshot save at {snap_path}")
    print("Segmentation Done")
