ngc batch run \
--name "ml-model.dsno-c256-sd64-data-0" --preempt RUNONCE \
--commandline "cd /DSNO-code; cd taming-transformers; pip3 install -e .;\
cd ../stable-diffusion; git config --global --add safe.directory /DSNO-code/stable-diffusion; git pull; sleep 2; \
pip3 install git+https://github.com/openai/CLIP.git; \
pip3 install transformers kornia; \
pip3 install omegaconf pytorch-lightning torch-fidelity einops lmdb; \
python3 generate.py --num_imgs=512000 --outdir=data/c256-sd64-db-0 --batchsize=64 --num_gpus=8 --num_steps=64 --save_step=8 --guidance=1.5 --seed=0" \
--image "nvidia/pytorch:22.12-py3" --ace nv-us-west-2 --instance dgx1v.16g.8.norm \
--result /results --workspace T3_Ez80ySDWgSGnN26jtOg:/DSNO-code --port 6006 --port 1234 --port 8888

ngc batch run \
--name "ml-model.exempt-dsno-c256-sd64-data-1" --preempt RUNONCE \
--commandline "cd /DSNO-code; cd taming-transformers; pip3 install -e .;\
cd ../stable-diffusion; git config --global --add safe.directory /DSNO-code/stable-diffusion; git pull; sleep 2; \
pip3 install git+https://github.com/openai/CLIP.git; \
pip3 install transformers kornia; \
pip3 install omegaconf pytorch-lightning torch-fidelity einops lmdb; \
python3 generate.py --num_imgs=512000 --outdir=data/c256-sd64-db-1 --batchsize=64 --num_gpus=8 --num_steps=64 --save_step=8 --guidance=1.5 --seed=10" \
--image "nvidia/pytorch:22.12-py3" --ace nv-us-west-2 --instance dgx1v.16g.8.norm \
--result /results --workspace T3_Ez80ySDWgSGnN26jtOg:/DSNO-code --port 6006 --port 1234 --port 8888

ngc batch run \
--name "ml-model.exempt-dsno-c256-sd64-data-2" --preempt RUNONCE \
--commandline "cd /DSNO-code; cd taming-transformers; pip3 install -e .;\
cd ../stable-diffusion; git config --global --add safe.directory /DSNO-code/stable-diffusion; git pull; sleep 2; \
pip3 install git+https://github.com/openai/CLIP.git; \
pip3 install transformers kornia; \
pip3 install omegaconf pytorch-lightning torch-fidelity einops lmdb; \
python3 generate.py --num_imgs=512000 --outdir=data/c256-sd64-db-2 --batchsize=64 --num_gpus=8 --num_steps=64 --save_step=8 --guidance=1.5 --seed 20" \
--image "nvidia/pytorch:22.12-py3" --ace nv-us-west-2 --instance dgx1v.16g.8.norm \
--result /results --workspace T3_Ez80ySDWgSGnN26jtOg:/DSNO-code --port 6006 --port 1234 --port 8888

ngc batch run \
--name "ml-model.exempt-dsno-c256-sd64-data-3" --preempt RUNONCE \
--commandline "cd /DSNO-code; cd taming-transformers; pip3 install -e .;\
cd ../stable-diffusion; git config --global --add safe.directory /DSNO-code/stable-diffusion; git pull; sleep 2; \
pip3 install git+https://github.com/openai/CLIP.git; \
pip3 install transformers kornia; \
pip3 install omegaconf pytorch-lightning torch-fidelity einops lmdb; \
python3 generate.py --num_imgs=512000 --outdir=data/c256-sd64-db-3 --batchsize=64 --num_gpus=8 --num_steps=64 --save_step=8 --guidance=1.5 --seed 30" \
--image "nvidia/pytorch:22.12-py3" --ace nv-us-west-2 --instance dgx1v.16g.8.norm \
--result /results --workspace T3_Ez80ySDWgSGnN26jtOg:/DSNO-code --port 6006 --port 1234 --port 8888

ngc batch run \
--name "ml-model.exempt-dsno-c256-sd64-data-4" --preempt RUNONCE \
--commandline "cd /DSNO-code; cd taming-transformers; pip3 install -e .;\
cd ../stable-diffusion; git config --global --add safe.directory /DSNO-code/stable-diffusion; git pull; sleep 2; \
pip3 install git+https://github.com/openai/CLIP.git; \
pip3 install transformers kornia; \
pip3 install omegaconf pytorch-lightning torch-fidelity einops lmdb; \
python3 generate.py --num_imgs=512000 --outdir=data/c256-sd64-db-4 --batchsize=64 --num_gpus=8 --num_steps=64 --save_step=8 --guidance=1.5 --seed 40" \
--image "nvidia/pytorch:22.12-py3" --ace nv-us-west-2 --instance dgx1v.16g.8.norm \
--result /results --workspace T3_Ez80ySDWgSGnN26jtOg:/DSNO-code --port 6006 --port 1234 --port 8888

ngc batch run \
--name "ml-model.exempt-dsno-c256-sd64-data-5" --preempt RUNONCE \
--commandline "cd /DSNO-code; cd taming-transformers; pip3 install -e .;\
cd ../stable-diffusion; git config --global --add safe.directory /DSNO-code/stable-diffusion; git pull; sleep 2; \
pip3 install git+https://github.com/openai/CLIP.git; \
pip3 install transformers kornia; \
pip3 install omegaconf pytorch-lightning torch-fidelity einops lmdb; \
python3 generate.py --num_imgs=512000 --outdir=data/c256-sd64-db-5 --batchsize=64 --num_gpus=8 --num_steps=64 --save_step=8 --guidance=1.5 --seed 50" \
--image "nvidia/pytorch:22.12-py3" --ace nv-us-west-2 --instance dgx1v.16g.8.norm \
--result /results --workspace T3_Ez80ySDWgSGnN26jtOg:/DSNO-code --port 6006 --port 1234 --port 8888

ngc batch run \
--name "ml-model.exempt-dsno-c256-sd64-data-6" --preempt RUNONCE \
--commandline "cd /DSNO-code; cd taming-transformers; pip3 install -e .;\
cd ../stable-diffusion; git config --global --add safe.directory /DSNO-code/stable-diffusion; git pull; sleep 2; \
pip3 install git+https://github.com/openai/CLIP.git; \
pip3 install transformers kornia; \
pip3 install omegaconf pytorch-lightning torch-fidelity einops lmdb; \
python3 generate.py --num_imgs=512000 --outdir=data/c256-sd64-db-6 --batchsize=64 --num_gpus=8 --num_steps=64 --save_step=8 --guidance=1.5 --seed 60" \
--image "nvidia/pytorch:22.12-py3" --ace nv-us-west-2 --instance dgx1v.16g.8.norm \
--result /results --workspace T3_Ez80ySDWgSGnN26jtOg:/DSNO-code --port 6006 --port 1234 --port 8888

ngc batch run \
--name "ml-model.exempt-dsno-c256-sd64-data-7" --preempt RUNONCE \
--commandline "cd /DSNO-code; cd taming-transformers; pip3 install -e .;\
cd ../stable-diffusion; git config --global --add safe.directory /DSNO-code/stable-diffusion; git pull; sleep 2; \
pip3 install git+https://github.com/openai/CLIP.git; \
pip3 install transformers kornia; \
pip3 install omegaconf pytorch-lightning torch-fidelity einops lmdb; \
python3 generate.py --num_imgs=512000 --outdir=data/c256-sd64-db-7 --batchsize=64 --num_gpus=8 --num_steps=64 --save_step=8 --guidance=1.5 --seed 70" \
--image "nvidia/pytorch:22.12-py3" --ace nv-us-west-2 --instance dgx1v.16g.8.norm \
--result /results --workspace T3_Ez80ySDWgSGnN26jtOg:/DSNO-code --port 6006 --port 1234 --port 8888
