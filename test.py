import sys
sys.dont_write_bytecode = True

import torch
import yaml
import wandb
import numpy as np
from pathlib import Path
from model.cvae import CVAE

import matplotlib.pyplot as plt
from matplotlib import cm
from torch.autograd import Variable

import copy


from data . dataloader import MNIST_Loader

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_yml(yml_path):
    with open(yml_path) as tyaml:
        yml = yaml.safe_load(tyaml)
        return yml

def one_hot_vector(labels,class_size):
    targets_id = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets_id[i, label] = 1
    return Variable(targets_id)

def show_imgs(data):
    data = np.asarray(data * 255, dtype=np.uint8)
    plt.figure("title", figsize=(len(data), 1))
    for i in range(len(data)):
        plt.subplot(1, len(data), i+1)
        plt.axis('off')
        plt.imshow(data[i].transpose(1, 2, 0), cmap=cm.gray, interpolation='nearest')
    plt.pause(100)
    plt.close()

def show_images_square(data, title='no title', n_data_max=100, n_data_per_row=10, save_fig=False):
    data = np.asarray(data * 255, dtype=np.uint8)
    n_data_total = min(data.shape[0], n_data_max) # 保存するデータの総数
    n_rows = n_data_total // n_data_per_row # 保存先画像においてデータを何行に分けて表示するか
    if n_data_total % n_data_per_row != 0:
        n_rows += 1
    plt.figure(title, figsize=(n_data_per_row, n_rows))
    for i in range(0, n_data_total):
        plt.subplot(n_rows, n_data_per_row, i+1)
        plt.axis('off')
        plt.imshow(data[i].transpose(1, 2, 0), cmap=cm.gray, interpolation='nearest')
    plt.pause(10)
    if save_fig:
        plt.savefig(title + '.png', bbox_inches='tight')
    plt.close()

def evals(conf):
    image_size = conf['image_size']
    h_dim = conf['h_dim']
    z_dim = conf['z_dim']
    cond_dim = conf['cond_dim']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    BASE_DIR = Path('.')

    WEIGHT_DIR = BASE_DIR / conf['weight']
    model_path = WEIGHT_DIR / (conf['model_name']+'.pth')

    model = CVAE(image_size,h_dim,z_dim,cond_dim).to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    with torch.no_grad():
        loader = MNIST_Loader(conf).__iter__()
        data, labels = loader.next()
        show_imgs(data[labels==3])
        x = data.to(device).view(-1, image_size).to(torch.float32)
        cond = one_hot_vector(labels,cond_dim)
        _,_,z = model.encoder(x,cond)
        out = model.decoder(z,cond)
        out = out[labels == 3]

        orig_z = copy.deepcopy(z)
        
        orig_outs = []
        for i in range(6):
            orig_out = model.decoder(orig_z,cond)[labels==0]
            orig_out = orig_out.view(-1,1,28,28)
            orig_out = orig_out.cpu().detach().numpy()
            orig_out = np.array(orig_out)

            orig_outs.append(orig_out)

            cond[labels == 0,1] = i * 1./5

        # print(np.concatenate(orig_outs,axis=0).shape)
        orig_outs = np.concatenate(orig_outs,axis=0)
        show_images_square(orig_outs,n_data_max=orig_outs.shape[0],n_data_per_row=orig_outs.shape[0]//6)

    out = out.view(-1,1,28,28)
    out = out.cpu().detach().numpy()

    return z, out

def main():
    config = load_yml('yml/test.yml')
    # wandb.init(project=config['project_name'], config=config, name=config['test_name'])
    # config = wandb.config

    torch.backends.cudnn.deterministic = True
    fix_seed(config['seed'])

    BASE_DIR = Path('.')
    WEIGHT_DIR = BASE_DIR
    TEST_YML = BASE_DIR

    WEIGHT_DIR = WEIGHT_DIR / config['weight']
    TEST_YML = TEST_YML / config['yml']

    z, out = evals(conf=config)
    # for i in range(len(out)):
    #     output_images = wandb.Image(out[i],caption=str(z[i]))
    #     wandb.log({'output':output_images})

    show_imgs(out)


if __name__ == '__main__':
    main()