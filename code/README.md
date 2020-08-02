# DSM-Net Experiments
This folder includes the DSM-Net experiments for VAE reconstruction and VAE generation using both box-shape and point cloud representations.

## Before start
To train the models, please first go to `data/partnetdata/`, `data/part_trees/` folder and download the training data.
To test over the pretrained models, please go to `data/models/` folder and download the pretrained checkpoints.

## Dependencies
This code has been tested on Ubuntu 18.04 with Cuda 10.0, GCC 5.5.0, Python 3.6.5, PyTorch 1.3.0, Jupyter IPython Notebook 5.7.8.

Please run
    
    pip3 install -r requirements.txt

to install the other dependencies.


## VAE Reconstruction & Generation & Interpolation
To train the network from scratch, run 

    bash scripts/train_vae_chair.sh

To test the model, run

    bash scripts/eval_vae_chair.sh

After running this script, the results of shape reconstructiona, generation and interpolation will be stored in the folder `data/results/box_vae_chair`

You can use `vis_pc.ipynb` to visualize the reconstruction results without rendered faces.


