module load cray-python
module load LUMI/22.08 partition/G rocm/5.1.4
source venv/bin/activate

export TRANSFORMERS_CACHE=$PWD/cache
export HF_DATASETS_CACHE=$PWD/cache
export HF_MODULES_CACHE=$PWD/cache
