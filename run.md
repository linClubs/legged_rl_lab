~~~python

 conda create -n lab3.0 python=3.12
  conda activate lab3.0
pip install isaaclab[isaacsim,all]==3.0.0b2 --extra-index-url https://pypi.nvidia.com

pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu124


python scripts/rsl_rl/train.py --task go2 --headless
python scripts/rsl_rl/train.py --task go2_handstand --headless


python scripts/rsl_rl/train.py --task go2 --headless --physics newton_mjwarp
~~~