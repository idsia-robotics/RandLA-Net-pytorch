from random import random

import numpy as np
from torch.utils import data


class RandlanetWeightedSampler(data.Sampler):

    def __init__(self, dataset, n_steps):
        self.weights = dataset.total_class_weight
        self.kdtrees = dataset.kdtrees
        self.labels = dataset.labels
        self.possibility = dict()
        self.min_possibility = dict()
        self.n_steps = n_steps
        self.cfg = dataset.cfg

    def __iter__(self):
        np.random.seed()
        for pc_id, kdtree in self.kdtrees.items():
            self.possibility[pc_id] = np.random.rand(len(kdtree.data)) * 1e-3
            self.min_possibility[pc_id] = float(np.min(self.possibility[pc_id]))

        for _ in range(self.n_steps):
            pc_id = min(self.min_possibility, key=self.min_possibility.get)
            point_idx = np.argmin(self.possibility[pc_id])
            # Get all points within the cloud from tree structure
            points = np.array(self.kdtrees[pc_id].data, copy=False)

            # Center point of input region
            center_point = points[point_idx, :].reshape(1, -1)

            # Add noise to the center point
            noise = np.random.normal(scale=self.cfg['noise_init'] / 10,
                                     size=center_point.shape)
            pick_point = center_point + noise.astype(center_point.dtype)
            # takes the indices of num_points neighbours
            query_idx = self.kdtrees[pc_id].query(pick_point,
                                                  k=self.cfg['num_points'])[1][0]
            # Get corresponding points and colors based on the index
            queried_pc_labels = self.labels[pc_id][query_idx]
            queried_pt_weight = np.array(
                [self.weights[lbl] for lbl in queried_pc_labels])
            # Update the possibility of the selected points
            dists = np.sum(
                np.square((points[query_idx] - pick_point).astype(np.float32)),
                axis=1)
            delta = np.square(1 - dists / np.max(dists)) * queried_pt_weight
            self.possibility[pc_id][query_idx] += delta
            self.min_possibility[pc_id] = float(np.min(self.possibility[pc_id]))
            yield pc_id, pick_point

    def __len__(self):
        return self.n_steps
