import sys
sys.dont_write_bytecode = True

import torch
from data . dataloader import MNIST_Loader
from model.vae import VAE
import torch.nn.functional as F
import yaml
import numpy as np
from pathlib import Path
import wandb

BASE_DIR = Path('.')
WEIGHT_DIR = BASE_DIR
TRAIN_YML = BASE_DIR

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_yml(yml_path):
    with open(yml_path) as tyaml:
        yml = yaml.safe_load(tyaml)
        return yml

def loss_function(label, predict, mu, log_var):
    reconstruction_loss = F.binary_cross_entropy(predict, label, reduction='sum')
    kl_loss = -0.5 * torch.sum(1+ log_var - mu.pow(2) - log_var.exp())
    vae_loss = reconstruction_loss + kl_loss
    return vae_loss, reconstruction_loss, kl_loss

def train(conf):
    epochs = conf['epochs']
    image_size = conf['image_size']
    h_dim = conf['h_dim']
    z_dim = conf['z_dim']
    lr = float(conf['learning_rate'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = VAE(image_size, h_dim, z_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    losses = []
    model.train()
    loader = MNIST_Loader(conf)

    for epoch in range(epochs):

        for i, (x, labels) in enumerate(loader):

            x = x.to(device).view(-1, image_size).to(torch.float32)
            x_recon, mu, log_var, z = model(x)
            loss, recon_loss, kl_loss = loss_function(x, x_recon, mu, log_var)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if (i+1) % 10 == 0:
                wandb.log({'epoch':epoch+1, 'loss':loss, 'recon_loss':recon_loss, 'kl_loss':kl_loss})

            losses.append(loss)
        wandb.save("vae.h5")

    return losses, model


def main():
    config  = load_yml('yml/train.yml')
    wandb.init(project=config['project_name'], config=config, name=config['train_name'])
    conf = wandb.config

    torch.backends.cudnn.deterministic = True
    fix_seed(conf['seed'])

    BASE_DIR = Path('.')
    WEIGHT_DIR = BASE_DIR
    TRAIN_YML = BASE_DIR

    WEIGHT_DIR = WEIGHT_DIR / conf['weight']
    if not WEIGHT_DIR.exists():
        WEIGHT_DIR.mkdir()
    TRAIN_YML = TRAIN_YML / conf['yml']

    _ , model = train(conf)

    model_path = WEIGHT_DIR / (conf['train_name']+'.pth')
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    main()