

# export PYTHONPATH=/media/eric/Data/projects/habitat-sim
# export PATH=/home/eric/anaconda3/envs/py365-hab/bin:/usr/bin
# export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
/media/eric/Data/nvidia/nsight-systems-2020.3.1/bin/nsys profile --sample=none --trace=nvtx --trace-fork-before-exec=true --debug-markers --gpuctxsw=false --delay=0 --stats=false --export=sqlite python habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ppo_pointnav_example.yaml --run-type train
# py-spy record -f speedscope -n -s -o ../profiles/pyspy_profile.speedscope -- habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ppo_pointnav_example.yaml --run-type train
# py-spy record --format speedscope -n -F -i -s -o "../profiles/pyspy_profile1.speedscope" -- habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ppo_pointnav_example.yaml --run-type train

