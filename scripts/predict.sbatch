#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 8GB
#SBATCH --time 8:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name=predict
#SBATCH --output logs/job.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bs3639@nyu.edu

CONFIG=$1
FILE=$2

../sing << EOF

python -m object_states.predict $CONFIG --file-path $FILE
python -m object_states.embed $CONFIG --file-path $FILE

EOF