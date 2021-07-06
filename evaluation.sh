#!/bin/bash
#BSUB -n12
#BSUB -R "rusage[ngpus_excl_p=0,mem=8192,scratch=20000]"
#BSUB -N
#BSUB -J msnovelist_eval
#BSUB -W 3:59
#BSUB -oo /msnovelist-data/logs/eval.%J.log
#BSUB -env "all"

RM_PICKLE=1

export MSNOVELIST_BASE=/msnovelist
EVAL_CONFIG_DIR=$WEIGHTS
TRAINED_CONFIG_DIR=$TRAINED

# For now we need this:
BASELINE_DIR=/msnovelist-data/baseline
EVAL_DIR=/msnovelist-data/eval

export COMPUTERNAME=DOCKER
export TF_CPP_MIN_LOG_LEVEL=3

UTAG=
if [[ "$TAG" != "" ]]
then
	UTAG=_$TAG
fi

LSB_JOBID=`date +%s`

COUNTER=0

eval "$(conda shell.bash hook)"
conda activate msnovelist-env

cd $MSNOVELIST_BASE

for TRAIN_CONFIG in `ls $TRAINED_CONFIG_DIR | grep .yaml$`
do

	for EVAL_CONFIG in `ls $EVAL_CONFIG_DIR | grep .yaml$`
	do
		((COUNTER++))
		echo "eval_id: '$LSB_JOBID'" > $TMPDIR/$LSB_JOBID.yaml
		echo "eval_counter: '$COUNTER'" >> $TMPDIR/$LSB_JOBID.yaml
		echo "fingerprinter_cache: /msnovelist-data/fingerprint-cache/fingerprint_cache.db" >> $TMPDIR/$LSB_JOBID.yaml
        	python "$MSNOVELIST_BASE/evaluation_mp.py" -c $TRAINED_CONFIG_DIR/$TRAIN_CONFIG $EVAL_CONFIG_DIR/$EVAL_CONFIG $TMPDIR/$LSB_JOBID.yaml
        	python "$MSNOVELIST_BASE/evaluation/identity_ranking_with_metrics.py" -c $TRAINED_CONFIG_DIR/$TRAIN_CONFIG $EVAL_CONFIG_DIR/$EVAL_CONFIG $TMPDIR/$LSB_JOBID.yaml 
        	python "$MSNOVELIST_BASE/evaluation/top_similarity_with_metrics.py" -c $TRAINED_CONFIG_DIR/$TRAIN_CONFIG $EVAL_CONFIG_DIR/$EVAL_CONFIG $TMPDIR/$LSB_JOBID.yaml
        	python "$MSNOVELIST_BASE/evaluation/top_rediscovery.py" -c $TRAINED_CONFIG_DIR/$TRAIN_CONFIG $EVAL_CONFIG_DIR/$EVAL_CONFIG $TMPDIR/$LSB_JOBID.yaml
		if [ "$RM_PICKLE" == "1" ]
		then
			rm ${EVAL_DIR}/eval_${LSB_JOBID}-*.pkl
		fi
	done
done

mkdir -p $TMPDIR/tarball/${TAG}
cp ${EVAL_DIR}/eval_${LSB_JOBID}_* $TMPDIR/tarball/${TAG}
tar cfz $TMPDIR/eval_${LSB_JOBID}${UTAG}.tar.gz -C $TMPDIR/tarball .
cp $TMPDIR/eval_${LSB_JOBID}${UTAG}.tar.gz ${EVAL_DIR}/

