from .unet import UNetPerturbationGenerator, UNetCBAMPerturbationGenerator, UNetDiffusion, UNetCBAMDiffusion

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

