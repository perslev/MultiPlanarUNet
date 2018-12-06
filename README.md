# Multi Planar UNet

This package implements fully autonomous deep learning based 
segmentation of any 3D medical image volume. It uses a fixed 
hyperparameter set and a fixed model topology, eliminating the need for
conducting hyperparameter tuning experiments.

This software may be used as-is and does not require deep learning expertise to
get started. It may also serve as a strong baseline method for general purpose
semantic segmentation of medical images.

The system has been evaluated on a wide range of tasks spanning organ and 
pathology segmentation across tissue types and scanner modalities. 
The model obtained a top-5 position at the 2018 Medical Segmentation Decathlon 
(http://medicaldecathlon.com/) despite its simplicity and computational 
efficiency.

## Method
The base model is a just slightly modified 2D UNet (https://arxiv.org/abs/1505.04597) 
trained under a multi-planar framework. Specifically, the 2D model is
fed images sampled across multiple views onto the image volume simultaneously:

[Multi Planar Animation](resources/multi_planar_training.gif)

At test-time, the model predict along each of the views and recreates a set of full segmentation volumes. 
These volumes are majority voted into one using a learned function that weights
each class from each view individually to maximuse the performance.

![](resources/multi_planar_model.png)

The method is described in detail below.
https://1drv.ms/b/s!AhgI3jIn2dNrhcp8yLYv0_EsC97u9A

## Usage

Project initialization, model training, evaluation, prediction etc. may be 
performed using the scripts located in ```MultiPlanarUNet.bin```. The script 
named ```mp.py``` serves as an entry point to all other scripts, and is used
as follows:

```bash
# Invoke the help menu
mp --help

# Launch the train script
mp train [arguments passed to 'train'...]
```

**Preparing the data**\
In order to train a model to solve a specific task, a set of manually 
annotated images must be stored in a folder under the following structure:

```
./data_folder/
|- train/
|--- images/
|------ image1.nii.gz
|------ image5.nii.gz
|--- labels/
|------ image1.nii.gz
|------ image5.nii.gz
|- val/
|--- images/
|--- labels/
|- test/
|--- images/
|--- labels/
|- aug/ <-- OPTIONAL
|--- images/
|--- labels/
```

The names of these folders may be customized in the parameter file (see below), 
but default to those shown above.

All images must be stored in the ``.nii``/```.nii.gz``` format. The image and 
corresponding label map files must be identically named.

The ```aug``` folder may store additional images that can be included during 
training with a lower weight assigned in optimization.

**Initializing a Project**\
Once the data is stored under the above folder structure, a Multi Planar 
project can be initialized as follows:

```
mp init_project --name my_project --data_dir ./data_folder
```

This will create a folder at path ```my_project``` and populate it with a YAML
file named ```train_hparams.yaml```, which stores all hyperparameters. Any 
parameter in this file may be specified manually, but can all be set 
automatically.

**Training**\
The model can now be trained as follows:

```
mp train --num_GPUs=2   # Any number of GPUs (or 0)
```

During training various information and images will be logged automatically to 
project folder. Typically, after training, the folder will look as follows:

```
./my_project/
|- images/               # Example segmentations through training
|- logs/                 # Various log files
|- model/                # Stores the best model parameters
|- tensorboard/          # TensorBoard graph and metric visualization
|- train_hparams.yaml    # The hyperparameters file
|- views.npz             # An array of the view vectors used
|- views.png             # Visualization of the views used
```

**Fusion Model Training**\
When using the MultiPlanar model, a fusion model must also be trained:
```
mp train_fusion --num_GPUs=2
```

**Predict and evaluate**\
The trained model can now be evaluated on the testing data in 
```data_folder/test```:

```
mp predict --num_GPUs=2 --out_dir predictions
```

This will create a folder ```my_project/predictions``` storing the predicted 
images along with dice coefficient performance metrics.

The model can also be used to predict on images without labels using the 
``--no_eval`` flag or on single files as follows:

```
mp predict -f ./new_image.nii.gz
```

