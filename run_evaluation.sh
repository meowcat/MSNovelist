#!/bin/bash

#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --account=es_biol
#SBATCH --mem-per-cpu=16G
#SBATCH --time=3:59:59
#SBATCH --tmp=4G

source .env

CMD="eval"
CMD_EXEC=""
OPTS=""

case $CMD in
	"eval")
		CMD_EXEC="/msnovelist/evaluation.sh"
		OPTS="--nv"
		FOLD=${SLURM_ARRAY_TASK_ID:-1}
		JOB=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
		;;
	"bash")
		CMD_EXEC="bash"
		OPTS=""
		FOLD=${SLURM_ARRAY_TASK_ID:-1}
		JOB=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
		;;
esac



# if [[ "$CMD" == "eval" ]]
# then
# 	mkdir -p $RESULTS_LOC/evaluation
# 	mkdir -p $RESULTS_LOC/weights
# 	echo "eval_id: '$JOB'" > $RESULTS_LOC/evaluation/$SLURM_JOB_ID.yaml
# 	echo "cv_fold: $FOLD" >> $RESULTS_LOC/evaluation/$SLURM_JOB_ID.yaml
# fi	
mkdir -p $RESULTS_LOC/evaluation
mkdir -p $RESULTS_LOC/weights
echo "eval_id: '$JOB'" > $RESULTS_LOC/evaluation/$SLURM_JOB_ID.yaml
echo "cv_fold: $FOLD" >> $RESULTS_LOC/evaluation/$SLURM_JOB_ID.yaml

echo "source _entrypoint.sh" >> $TMPDIR/.bashrc

cp $DATA_LOC/*.db $TMPDIR
cp $DATA_LOC/*.pkl $TMPDIR
cp $DATA_LOC/*.tsv $TMPDIR
cp $DATA_LOC/*.yaml $TMPDIR
cp $CODE_LOC/*.yaml $TMPDIR
cp -r $DATA_LOC/.sirius-6.0 $TMPDIR

singularity run \
	--bind $TMPDIR:/$HOME \
	--env-file .env \
	$SIF_LOC \
	/contrib/sirius/bin/sirius login \
	--user-env=SIRIUS_USERNAME \
	--password-env=SIRIUS_PASSWORD

singularity run \
	--bind $TMPDIR:/$HOME \
	--env-file .env
	$SIF_LOC \
	/contrib/sirius/bin/sirius login \
	--select-subscription="$SIRIUS_LICENSE"

singularity run \
	$OPTS \
	--bind $TMPDIR:/$HOME \
	--bind $TMPDIR:/sirius6_db \
	--bind $TMPDIR:/target \
	--bind $CODE_LOC:/msnovelist \
	--bind $RESULTS_LOC:/data \
	--bind $RESULTS_LOC:/msnovelist-data \
	$SIF_LOC \
	$CMD_EXEC

	