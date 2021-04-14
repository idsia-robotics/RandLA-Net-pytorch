# RandLA-Net-pytorch
Our PyTorch implementation of RandLA-Net (https://github.com/QingyongHu/RandLA-Net).

We tried to stay as close as possible to the original Tensorflow model implementation.
However, some changes in the pipeline and input format were made to adapt the model to our own data format.

In particular, each pointcloud must be stored in its own folder called "pc_id=numerical_id".
This folder must contain a pickle file ("pc.pickle") for the pointcloud itself and another folder called "metadata".
Finally, metadata folder must contain a file called "metadata.pickle", which contains a python dictionary like this:
{
    "pc_id": int id for the pointloud, 
    "labels": list of floats representing the classes available in the pointcloud, 
    "name": string name for the pointcloud (can be None)
}

Each "pc.pickle" file must contain a numpy array of shape (n, 7), where n is the number of points and 7 are (in this order):
- x,y,z float coordinates
- r, g, b colors which are integers in [0, 255] (stored, however, as floats)
- ground truth label

In the "data" folder a dataset examples is available (it contains two low density pointclouds taken from nomoko dataset, available here).
This can be used as reference to try training and testing the model.
Training will produce as output the model checkpoints (by default under the data folder).
Testing will produce the following files (inside the selected model folder):
    - xyz_tile.pickle: array of shape (n, 3) containing xyz coordinates
    - xyz_probs.pickle: array of shape (n, n_classes) containing the predicted score for each class
    - xyz_labels.pickle: array of shape (n,) containing the most probable class for each point (i.e. argmax(xyz_probs))
    - true_rgb.pickle: array of shape (n, 3) containing rgb colors
    - gt_labels.pickle: array of shape (n,) containing the ground truth class for each point
Also, it produces two visualization of the pointcloud to allow for quick inspection of results:
    - snapshot_predictions.html: pyKDL html snapshot to visualize the pointcloud colored according to model predictions (each predicted class can be toggled in the UI)
    - rgb_predictions.html: pyKDL html snapshot to visualize the pointcloud colored with real rgb (each predicted class can be toggled in the UI)

"model.py", "sampler.py" and "dataset.py" contain all the relevant pytorch code to be reused and adapted for different data formats.
"hyperparameters.py" contains the hyperparameters that can be set to train and test the model

# How to run
First, build and run singularity
from repository root folder:
singularity build --fakeroot RandLA-Net-pytorch.sif Definition.def
singularity instance start --nv RandLA-Net-pytorch.sif randlanet

Open a shell inside the container:
singularity shell instance://randlanet

From within the shell, run training:
python3 train.py

and testing:
python3 test.py