pip3 install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip3 install omegaconf>=2.0.0 pytorch-lightning>=1.0.8 torch-fidelity einops
python3 generate.py --num_imgs=512000 --outdir=data/c256-sd64-db-0 --batchsize=64 --num_gpus=8 --save_step=8 --guidance=1.5 --seed 0