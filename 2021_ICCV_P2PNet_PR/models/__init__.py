from .p2pnet import build
from .unet import UNetPerturbationGenerator, UNetCBAMPerturbationGenerator, UNetDiffusion, UNetCBAMDiffusion
# build the P2PNet model
# set training to 'True' during training
def build_model(args, training=False):
    return build(args, training)


def build_unet(args, type='vanilla'):
    if type=='vanilla':
        return UNetPerturbationGenerator() #Keep default params
    elif type=='attention':
        return UNetCBAMPerturbationGenerator() #Keep default params


def build_diffusion(args, type='vanilla'):
    if type=='vanilla':
        return UNetDiffusion() #Keep default params
    elif type=='attention':
        return UNetCBAMDiffusion() #Keep default params