import sys
sys.dont_write_bytecode = True

import torch
from data . dataloader import MNIST_Loader
from model.cvae import CVAE
import torch.nn.functional as F
import yaml
import numpy as np
from pathlib import Path
import wandb
from torch.autograd import Variable

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

def loss_function(correct, predict, mu, log_var):
    reconstruction_loss = F.binary_cross_entropy(predict, correct, reduction='sum')
    kl_loss = -0.5 * torch.sum(1+ log_var - mu.pow(2) - log_var.exp())
    cvae_loss = reconstruction_loss + kl_loss
    return cvae_loss, reconstruction_loss, kl_loss

def one_hot_vector(labels,class_size):
    targets_id = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets_id[i, label] = 1
    return Variable(targets_id)


def train(conf):
    epochs = conf['epochs']
    image_size = conf['image_size']
    h_dim = conf['h_dim']
    z_dim = conf['z_dim']
    lr = float(conf['learning_rate'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loader = MNIST_Loader(conf)
    cond_dim = loader.dataset.train_labels.unique().size(0)

    model = CVAE(image_size, h_dim, z_dim,cond_dim).to(device)
    if torch.cuda.is_available():
        model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    losses = []
    model.train()

    for epoch in range(epochs):

        for i, (x, labels) in enumerate(loader):        
            x = x.to(device).view(-1, image_size).to(torch.float32)
            cond = one_hot_vector(labels,cond_dim)
            x_recon, mu, log_var, z = model(x,cond)
            loss, recon_loss, kl_loss = loss_function(x, x_recon, mu, log_var)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if (i+1) % 10 == 0:
                wandb.log({'epoch':epoch+1, 'loss':loss, 'recon_loss':recon_loss, 'kl_loss':kl_loss})

            losses.append(loss)
        wandb.save("cvae.h5")

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