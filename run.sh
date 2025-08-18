#SBATCH --job-name=parser

#SBATCH --mail-type=ALL 

#SBATCH --mail-user=dayneguy@usf.edu 

#SBATCH --output=/home/d/dayneguy/gaivi_output/parser.%j 
#SBATCH -e=/home/d/dayneguy/gaivi_output/parser-err.%j

#SBATCH --gpus=4

#SBATCH --time=14:00:00 

#SBATCH --mem=100GB 

nvidia-smi 

. /apps/anaconda3/etc/profile.d/conda.sh 

conda activate local-llm 

cd /data/dayneguy/parser

python3 hyperparam_search.py
