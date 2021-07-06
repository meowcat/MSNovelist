#!/bin/bash

DIRS=`ls -d */`
WEIGHTS_DIR=/msnovelist-data/weights
OUT_PREFIX=config/weights_fold

# This script finds config.yaml files for models, written during training,
# and modifies them for direct use in the evaluation script.
# It removes:
# * fingerprint cache, database, and baseline paths
# * potentially set eval_counter/id
# * evaluation_set and eval_detail setting
# It adds
# * selection of an evaluation set and eval_detail=True
# * selection of a decoder
# It replaces the path to the original CSV eval database with the
#   replacement pickle, which includes the holdout set definition


for DIR in $DIRS
do
    echo checking $DIR
    TDIR=${OUT_PREFIX}${DIR}
    if [[ -f "${DIR}config.yaml" ]]
    then
        mkdir -p $TDIR
	RELDIR=`realpath --relative-to="${WEIGHTS_DIR}" $DIR`
	WEIGHTS=`ls $DIR | grep "w.*.hdf5"`
	for WEIGHT in $WEIGHTS
	do
		TARGET=`echo $WEIGHT | sed -E s/-[0-9][.].*//`
	 	echo "weights: $RELDIR/$WEIGHT" > $TDIR/$TARGET.yaml
	done
    fi
done

