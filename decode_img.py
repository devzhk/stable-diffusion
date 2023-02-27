import torch
import click
import os
from omegaconf import OmegaConf
from tqdm import tqdm
import lmdb

from ldm.util import instantiate_from_config

import numpy as np 
from PIL import Image


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model(config_path, ckpt_path):
    config = OmegaConf.load(config_path)  
    model = load_model_from_config(config, ckpt_path)
    return model


def save2dir(images, outdir, curr):
    num_imgs = images.shape[0]
    for j in range(num_imgs):
        im = Image.fromarray(images[j])
        img_path = os.path.join(outdir, f'{j + curr}.png')
        im.save(img_path)    
    return curr + num_imgs


def get_batch_from_db(env, num_entries, curr, data_shape):
    '''
    Fetch num_entries images from env
    '''
    batch_list = []
    with env.begin(write=False) as txn:
        for i in range(curr, curr + num_entries):
            key = f'{i}'.encode()
            value = txn.get(key)
            val = np.frombuffer(value, dtype=np.float32)
            data = val.reshape(data_shape)  # T, C, H, W
            batch_list.append(data[-1])
    batch = np.stack(batch_list, axis=0)    # num_entries, C, H, W
    return batch


@torch.no_grad()
def decode_db(model, num_imgs, root_dir, outdir, num_label_per_db=2, batchsize=32):
    # 
    os.makedirs(outdir, exist_ok=True)
    device = torch.device('cuda:0')
    data_shape = (9, 3, 64, 64)
    num_batches = num_imgs // batchsize

    db_dir = os.path.join(root_dir, 'lmdb')
    env = lmdb.open(db_dir, map_size=1000*1024*1024*1024, readahead=False)
    num_db_imgs = env.stat()['entries']
    num_imgs = min(num_imgs, num_db_imgs)
    # get labels 
    
    # if num_label_per_db > 1:
    label_list = []
    for i in range(num_label_per_db):
        label_path = os.path.join(root_dir, f'label-{i}.npy')
        sublabels = np.load(label_path)
        label_list.append(sublabels)
    labels = np.concatenate(label_list, axis=None)
    # else:
    #     label_path = os.path.join(root_dir, 'label-0.npy')
    #     labels = np.load(label_path)
    
    curr = 0
    # read latent codes
    for j in range(num_batches):
        print(f'labels of batch {j}: {labels[curr: curr + batchsize]}')
        # read entries from db
        batch = get_batch_from_db(env, batchsize, curr=j * batchsize, data_shape=data_shape)
        batch_th = torch.from_numpy(batch).to(device)
        imgs = model.decode_first_stage(batch_th)
        imgs = imgs.add_(1).mul(127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)    # B, C, H, W -> B, H, W, C
        img_arr = imgs.cpu().numpy()
        curr = save2dir(images=img_arr, outdir=outdir, curr=curr)




@click.command()
@click.option('--config', 'config_path', help='path to configuration file', type=str, required=True, default='configs/latent-diffusion/cin256-v2.yaml')
@click.option('--ckpt', 'ckpt_path', help='path to checkpoint', type=str, required=True, default='models/ldm/cin256-v2/model.ckpt')
@click.option('--num_imgs', help='Number of images to generate', metavar='INT', type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--outdir', help='Output directory', type=str, default='out', show_default=True)
@click.option('--dbdir', help='DB directory', type=str, default='data/c256-sd64-db0', show_default=True)
@click.option('--num_label_per_db', help='Number of labels files', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)

def main(config_path, ckpt_path, num_imgs, outdir, dbdir, num_label_per_db):
    model = get_model(config_path, ckpt_path)
    decode_db(model, num_imgs, dbdir, outdir, num_label_per_db=num_label_per_db, batchsize=100)


if __name__ == '__main__':
    main()