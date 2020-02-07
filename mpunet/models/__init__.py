from .unet import UNet
from .unet3D import UNet3D
from .fusion_model import FusionModel
from .model_init import model_initializer
from .multitask_unet2d import MultiTaskUNet2D

# Prepare a dictionary mapping from model names to data prep. functions
from mpunet.preprocessing import data_preparation_funcs as dpf

PREPARATION_FUNCS = {
    "UNet": dpf.prepare_for_multi_view_unet,
    "UNet3D": dpf.prepare_for_3d_unet,
    "MultiTaskUNet2D": dpf.prepare_for_multi_task_2d
}
