from .unet import UNet
from .unet3D import UNet3D
from .fusion_model import FusionModel
from .model_init import model_initializer
from .autofocus_unet import AutofocusUNet2D
from .st_unet3D import STUNet3D
from .st_unet3D_picewise import PieceWiseSTUNet3D


# Prepare a dictionary mapping from model names to data prep. functions
from MultiPlanarUNet.preprocessing import data_preparation_funcs as dpf

PREPARATION_FUNCS = {
    "UNet": dpf.prepare_for_multi_view_unet,
    "UNet3D": dpf.prepare_for_3d_unet,
    "AutofocusUNet2D": dpf.prepare_for_multi_view_unet,
    "STUNet3D": dpf.prepare_for_3d_unet,
    "PieceWiseSTUNet3D": dpf.prepare_for_3d_unet
}
