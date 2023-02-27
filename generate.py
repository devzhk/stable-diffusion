import torch
import click
import os
from omegaconf import OmegaConf
from tqdm import tqdm
import lmdb

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

import torch.multiprocessing as mp
import numpy as np 
from PIL import Image

from utils import download_model


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


def get_random_label(batchsize, num_classes, device):
    labels = torch.randint(low=0, high=num_classes, size=(batchsize, ), device=device)
    return labels


def save2db(traj_batch, env, curr):
    '''
    Input: 
        traj: ndarray, (B, T, C, H, W)
    '''
    num_traj = traj_batch.shape[0]
    with env.begin(write=True) as txn:
        for i in range(num_traj):
            key = f'{curr+i}'.encode()
            txn.put(key, traj_batch[i])
    return curr + num_traj


def save2dir(images, outdir, curr):
    num_imgs = images.shape[0]
    for j in range(num_imgs):
        im = Image.fromarray(images[j])
        img_path = os.path.join(outdir, f'{j + curr}.png')
        im.save(img_path)    
    return curr + num_imgs


@torch.no_grad()
def generate2png(model, sampler, 
                 num_samples, batch_size, outdir, 
                 num_steps=32, save_step=8,
                 scale=2.5, num_classes=1000, eta=0.0, 
                 start_idx=0, rank=0):
    '''
    - start_idx: index of the 1st image
    '''
    curr = start_idx
    num_batches = num_samples // batch_size
    
    db_dir = os.path.join(outdir, 'lmdb')
    os.makedirs(db_dir, exist_ok=True)
    label_path = os.path.join(outdir, f'label-{rank}.npy')
    env = lmdb.open(db_dir, map_size=1000*1024*1024*1024, readahead=False)
    labels = np.zeros((num_samples,))

    with model.ema_scope():
        # unconditional condition
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(batch_size*[num_classes]).to(model.device)}
            )
        for i in tqdm(range(num_batches), disable=(rank!=0)):
            xc = get_random_label(batchsize=batch_size, num_classes=num_classes, device=model.device)
            labels[i * batch_size: i * batch_size + batch_size] = xc.cpu().numpy()
            c = model.get_learned_conditioning({model.cond_stage_key: xc})
            samples, _ = sampler.sample_traj(S=num_steps,
                                            conditioning=c,
                                            batch_size=batch_size,
                                            shape=[3, 64, 64],
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc, 
                                            eta=eta, 
                                            log_every_t=save_step)
            # samples: T, B, C, H, W -> B, T, C, H, W
            traj = torch.stack(samples, dim=1).cpu().numpy() 
            curr = save2db(traj, env, curr)  
            # images = model.decode_first_stage(samples[-1])
            # images = images.add_(1).mul(127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)    # B, C, H, W -> B, H, W, C
            # img_arr = images.cpu().numpy()
            # curr = save2dir(images=img_arr, outdir=outdir, curr=curr)
    np.save(label_path, labels)
    print(f'{curr - start_idx} images has been generated by rank {rank}')


def subprocessfn(rank, config_path, ckpt_path, num_imgs, batchsize, outdir, num_steps, save_step, scale, seed):
    torch.cuda.set_device(rank)
    # setup model and sampler

    torch.manual_seed(seed+rank)
    model = get_model(config_path, ckpt_path)
    sampler = DPMSolverSampler(model)
    start_idx = num_imgs * rank
    print(num_imgs, batchsize)
    generate2png(model, sampler, 
                 num_imgs, batchsize, outdir, num_steps, save_step, scale, start_idx=start_idx, rank=rank)
    print(f'Rank {rank} exits.')


#---------------------------------
@click.command()
@click.option('--config', 'config_path', help='path to configuration file', type=str, required=True, default='configs/latent-diffusion/cin256-v2.yaml')
@click.option('--ckpt', 'ckpt_path', help='path to checkpoint', type=str, required=True, default='models/ldm/cin256-v2/model.ckpt')
@click.option('--num_imgs', help='Number of images to generate', metavar='INT', type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--batchsize', help='Batchsize', metavar='INT', type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--outdir', help='Output directory', type=str, default='out', show_default=True)
@click.option('--num_steps', help='Number of solver steps', metavar='INT', type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--save_step', help='Save data every save_step steps', metavar='INT', type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--seed', help='random seed', metavar='INT', type=click.IntRange(min=0), default=1, show_default=True)
@click.option('--guidance', help='Guidance scale', type=click.FloatRange(min=0), default=1.5, show_default=True)
@click.option('--num_gpus', help='Number of GPUs', type=click.IntRange(min=1), default=1, show_default=True)


#-----------------example--------------#
# python3 generate.py --num_imgs=50000 --outdir=out/g1.5dpm25 --guidance=1.5

def main(config_path, ckpt_path, num_imgs, batchsize, outdir, num_steps, save_step, seed, guidance, num_gpus):
    if not os.path.exists(ckpt_path):
        download_model('c256v2')

    os.makedirs(outdir, exist_ok=True)
    mp.set_start_method('spawn')
    # sampler = DDIMSampler(model)
    if num_gpus > 1:
        num_img_per_gpu = num_imgs // num_gpus
        processes = []
        for rank in range(num_gpus):
            print(f'Local rank {rank}')
            p = mp.Process(target=subprocessfn, args=(rank, config_path, ckpt_path, num_img_per_gpu, batchsize, outdir, num_steps, save_step, guidance, seed))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
    else:
        print('Running on single GPU')
        subprocessfn(0, config_path, ckpt_path, num_imgs, batchsize, outdir, num_steps, save_step, guidance, seed)


if __name__ == '__main__':
    main()