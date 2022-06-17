import sys
sys.dont_write_bytecode = True

import torch
import yaml
import wandb
import numpy as np
from pathlib import Path
from model.vae import VAE

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_yml(yml_path):
    with open(yml_path) as tyaml:
        yml = yaml.safe_load(tyaml)
        return yml

def evals(conf):
    image_size = conf['image_size']
    h_dim = conf['h_dim']
    z_dim = conf['z_dim']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    BASE_DIR = Path('.')

    WEIGHT_DIR = BASE_DIR / conf['weight']
    model_path = WEIGHT_DIR / (conf['model_name']+'.pth')

    model = VAE(image_size,h_dim,z_dim).to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    with torch.no_grad():
        z = torch.randn(25, z_dim).to(device)
        out = model.decoder(z)

    out = out.view(-1,28,28)
    out = out.cpu().detach().numpy()

    return z, out

def main():
    config = load_yml('yml/test.yml')
    wandb.init(project=config['project_name'], config=config, name=config['test_name'])
    conf = wandb.config

    torch.backends.cudnn.deterministic = True
    fix_seed(conf['seed'])

    BASE_DIR = Path('.')
    WEIGHT_DIR = BASE_DIR
    TEST_YML = BASE_DIR

    WEIGHT_DIR = WEIGHT_DIR / conf['weight']
    TEST_YML = TEST_YML / conf['yml']

    z, out = evals(conf=conf)
    for i in range(len(out)):
        output_images = wandb.Image(out[i],caption=str(z[i]))
        wandb.log({'output':output_images})


if __name__ == '__main__':
    main()