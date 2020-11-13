#!/bin/bash
trap 'catch $? $LINENO' EXIT
catch() {
  if [ "$1" != "0" ]; then
    echo "An error occured during the installation of Habitat-sim or Habitat-API." >&2
  fi
}
set -e
shopt -s extglob

#Checks if in Google Colab and does not run installation if it is not
python -c 'import google.colab' 2>/dev/null || (echo "Google Colab not detected: skipping installation" >&2  &&  exit)
#Don't run again if it's already installed
[ -f /content/habitat_sim_installed ] && exit
#Install Miniconda
cd /content/
wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -bfp /usr/local

#Adds the conda libraries directly to the colab path.
ln -s /usr/local/lib/python3.6/dist-packages /usr/local/lib/python3.6/site-packages

##Install Habitat-Sim and Magnum binaries
conda config --set default_threads 4 #Enables multithread conda installation
conda install -y --prefix /usr/local -c aihabitat -c conda-forge bullet=2.88 habitat-sim=0.1.6 headless withbullet python=3.6

#Shallow GIT clone for speed
# use my fork and profiling_tutorial branch
git clone https://github.com/eundersander/habitat-api --depth 20 --branch eundersander/profiling_tutorial

#Install Requirements.
cd /content/habitat-api/
set +e
python3.6 -m pip install -r /content/habitat-api/requirements.txt
reqs=(/content/habitat-api/habitat_baselines/**/requirements.txt)
python3.6 -m pip install "${reqs[@]/#/-r}"
python3.6 setup.py develop --all
set -e

#Download Assets
wget -c http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip && unzip -o habitat-test-scenes.zip
#wget -c http://dl.fbaipublicfiles.com/habitat/objects_v0.1.zip && unzip -o objects_v0.1.zip -d data/objects/
#wget -c http://dl.fbaipublicfiles.com/habitat/locobot_merged.zip && unzip -o locobot_merged.zip -d data/objects

touch /content/habitat_sim_installed
