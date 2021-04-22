# RandLA-Net-pytorch
[![Resuts example viz](https://github.com/idsia-robotics/RandLA-Net-pytorch/blob/main/RandLA-Net-pytorch_Visualization.gif)](https://youtu.be/qE3vvh8aY00)

Our PyTorch implementation of [RandLA-Net](https://github.com/QingyongHu/RandLA-Net)

We tried to stay as close as possible to the original Tensorflow model implementation.
However, some changes in the pipeline and input format were made to adapt the model to our own data format.

[model.py](https://github.com/idsia-robotics/RandLA-Net-pytorch/blob/main/model/model.py), [sampler.py](https://github.com/idsia-robotics/RandLA-Net-pytorch/blob/main/model/sampler.py), and [dataset.py](https://github.com/idsia-robotics/RandLA-Net-pytorch/blob/main/model/dataset.py) contain all the relevant pytorch code to be reused and adapted for different data formats. [hyperparameters.py](https://github.com/idsia-robotics/RandLA-Net-pytorch/blob/main/model/hyperparameters.py) contains... the hyperparameters that can be set for training.

Instructions are provided to run the complete pipeline on a data sample and to explain how to prepare your own data.

## Input Format
 Each pointcloud must be stored in its own folder named **pc_id=integer_id**.
This folder must contain a pickle file (**pc.pickle**) for the pointcloud itself and another folder called **metadata**.
Finally, metadata folder must contain a file called **metadata.pickle**, which contains a python dictionary like this:
  
	{
	    "pc_id": int id for the pointloud, 
	    "labels": list of floats representing the classes available in the pointcloud, 
	    "name": string name for the pointcloud (can be None)
	}
Here is an example of how the dataset folder would look like:
```
data/
	|________pc_id=20/
	|		|________pc.pickle
	|		|________metadata/
	|				|________metadata.pickle
	|________pc_id=30/
			|________pc.pickle
			|________metadata/
					|________metadata.pickle
```
Each **pc.pickle** file must contain a numpy array of shape *(n, 7)*, where *n* is the number of points and 7 are (in this order):
- x,y,z float coordinates
- r, g, b colors which are integers in [0, 255] (stored, however, as floats)
- ground truth label
In the [data](https://github.com/idsia-robotics/RandLA-Net-pytorch/tree/main/data) folder a sample dataset is available. It contains two low density pointclouds from [Nomoko](https://nomoko.world/) dataset, available [here](https://zenodo.org/record/4390295#.YIEin3UzY5k).
## Output Format
Training will produce as output pytorch model checkpoints (by default under the data folder).

Testing will produce the following files (inside the selected model folder):
   - **xyz_tile.pickle**: array of shape (n, 3) containing xyz coordinates
   - **xyz_probs.pickle**: array of shape (n, n_classes) containing the predicted score for each class
   - **xyz_labels.pickle**: array of shape (n,) containing the most probable class for each point (i.e. argmax(xyz_probs))
   - **true_rgb.pickle**: array of shape (n, 3) containing rgb colors
   - **gt_labels.pickle**: array of shape (n,) containing the ground truth class for each point

Also, it produces two visualization of the pointcloud to allow for quick inspection of results:
   - **snapshot_predictions.html**: K3D html snapshot to visualize the pointcloud colored according to model predictions (each predicted class can be toggled in the UI)
   - **rgb_predictions.html**: K3D html snapshot to visualize the pointcloud colored with real rgb (each predicted class can be toggled in the UI)

## How To Run
We provide a [singularity definition](https://github.com/idsia-robotics/RandLA-Net-pytorch/blob/main/Definition.def) to build the needed environment.
Please, note that if you store your dataset in another folder, you will need to change the **DATA_ROOT_PATH** variable in [utils.py](https://github.com/idsia-robotics/RandLA-Net-pytorch/blob/main/model/utils.py). Default points to *data/* within this repo.

```
# Clone repo and cd into it
git clone https://github.com/idsia-robotics/RandLA-Net-pytorch.git
cd RandLA-Net-pytorch

# Build singularity container
singularity build --fakeroot RandLA-Net-pytorch.sif Definition.def

# Start the singularity container
singularity instance start --nv RandLA-Net-pytorch.sif randlanet

# Open a shell inside the container:
singularity shell instance://randlanet

# From within the shell, run training:
python3 train.py

# and testing:
python3 test.py
```
# Related

####  Swiss3DCities: Aerial Photogrammetric 3D Pointcloud Dataset with Semantic Labels
   - [paper](https://arxiv.org/abs/2012.12996)
   - [dataset](https://zenodo.org/record/4390295) 
