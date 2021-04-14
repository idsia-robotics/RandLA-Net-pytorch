from model.testing import segment_randlanet
from model.hyperparameters import hyp
from model.dataset import RandlanetDataset
from model.training import train_randlanet_model
# media/gabri/ext_ssd/nomoko
# train_set = RandlanetDataset(["/media/gabri/ext_ssd/nomoko/datasets/full_pc/pc_id=39/"], **hyp)
train_randlanet_model(train_set_list =["data/pc_id=636/"],
                      test_set_list = ["data/pc_id=637/"],
                      hyperpars=hyp,
                      use_mlflow=False,
                      num_workers=4,
                      model_name="repo_example")


# segment_randlanet("/data/saved_models/model_randlanet_all_pc_lr_sched/",
#                   f"/data/datasets/full_pc/pc_id=50/",
#                   hyp,
#                   8, segmentation_name=f'all_pc_50')
# import pickle
# import torch
# import glob
# from segmentation.deep_learning.randlanet.model.torch_model import RandlaNet
# from segmentation.deep_learning.randlanet.model.torch_dataset import RandlanetDataset
# from segmentation.deep_learning.randlanet.model.torch_sampler import (
#     RandlanetWeightedSampler,
# )
# from segmentation.utils.path_utils import MODEL_SAVES_PATH
# from torch.utils import data
# from segmentation.deep_learning.randlanet.model.asegmentation import (
#     segment,
#     store_results,
# )
# from segmentation.utils.k3d_utils import generate_k3d_plot, color_mapping, name_mapping


# model_name = "all_pc_lr_sched"
# pc_path = "/data/datasets/full_pc/pc_id=50/"
# pc_id = 50
# model_root_folder = MODEL_SAVES_PATH + f"model_randlanet_" f"{model_name}/"
# with open(f"{model_root_folder}output/metadata.pkl", "rb") as f:
#     metadata = pickle.load(f)
# mapping = metadata["label_mapping"]
# best_epoch = metadata["best_epoch"]

# path_to_model = glob.glob(f"{model_root_folder}checkpoints/{best_epoch}_*.pth")[0]
# print(f"Loading model checkpoint: {model_name}")
# inv_mapping = {mapping[l]: l for l in mapping}
# print(f"Label inverse mapping: {inv_mapping}")

# n_classes = len(mapping)
# print(f"Best epoch was {metadata['best_epoch']}")
# print("Setting up pytorch")
# use_cuda = torch.cuda.is_available()
# print(f"Use cuda: {use_cuda}")
# device = torch.device("cuda:1" if use_cuda else "cpu")


# test_params = {
#     "batch_size": hyp["val_batch_size"],
#     "shuffle": False,
#     "num_workers": 4
# }


# test_set = RandlanetDataset([pc_path], **hyp)
# test_sampler = RandlanetWeightedSampler(
#     test_set, hyp["val_batch_size"] * hyp["val_steps"]
# )

# test_loader = data.DataLoader(test_set, sampler=test_sampler, **test_params)
# model = RandlaNet(
#     n_layers=hyp["num_layers"], n_classes=hyp["num_classes"], d_out=hyp["d_out"]
# )

# if device=="cpu":
#     map_location = torch.device("cpu")
#     model.load_state_dict(torch.load(path_to_model, map_location=map_location))
# else:
#     model.load_state_dict(torch.load(path_to_model))
# model = model.to(device)

# # segment tiles, tile by tile, classifying samples in batches of chunk_size
# # sampling each single point
# xyz_tile, xyz_labels, xyz_probs, true_rgb, gt_labels = segment(
#     test_loader, model, device, inv_mapping, hyp, 150
# )
# store_results(
#     model_root_folder,
#     xyz_tile,
#     xyz_labels,
#     xyz_probs,
#     true_rgb,
#     gt_labels,
#     pc_path,
#     f"all_pc_{pc_id}"
# )
# mask_map = {}
# for label in mapping.values():
#     mask = xyz_labels == label
#     mask_map[label] = mask

# plot = generate_k3d_plot(xyz_tile, mask_map=mask_map, mask_color=color_mapping, name_map=name_mapping)
# snapshot = plot.get_snapshot(9)
# snap_path = f"{model_root_folder}output/segmentations/all_pc_{pc_id}/predictions.html"
# with open(snap_path, 'w') as fp:
#     fp.write(snapshot)
#     print(f"Snapshot save at    {snap_path}")

# plot = generate_k3d_plot(xyz_tile, rgb=true_rgb, mask_map=mask_map, name_map=name_mapping)
# snapshot = plot.get_snapshot(9)
# snap_path = f"{model_root_folder}output/segmentations/all_pc_{pc_id}/rgb.html"
# with open(snap_path, 'w') as fp:
#     fp.write(snapshot)
#     print(f"Snapshot save at    {snap_path}")

# print("Segmentation Done")
