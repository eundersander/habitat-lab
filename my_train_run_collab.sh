

# export PYTHONPATH=/media/eric/Data/projects/habitat-sim
# export PATH=/home/eric/anaconda3/envs/py365-hab/bin:/usr/bin
# export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
echo "--trace=cuda,cudnn,opengl,nvtx,osrt"
/content/nsight_systems/bin/nsys profile --sample=none --trace=cuda,cudnn,opengl,nvtx,osrt --trace-fork-before-exec=true --debug-markers --gpuctxsw=false --delay=0 --stats=false --export=sqlite python habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ppo_pointnav_example.yaml --run-type train

