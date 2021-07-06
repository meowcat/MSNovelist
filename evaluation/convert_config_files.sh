#!/bin/bash

DIRS=`ls -d */`

# arguments are
# $1 top-k for evaluation, if empty this is 128
# $2 the target tag for the model; if empty it is left unchanged
# $3 the evaluation set; if empty it is "sirius"
# $4 the selected fold. If "X", then all folds are selected
# $5 the selected decoder, beam_search usually (set to stochastic_sampler for sampling)

# alternative for "sirius" is "casmi"
EVALUATION_SET=${3:-sirius}
EVAL_K=${1:-128}
EVAL_KK=${1:-128}
TAG=$2
DECODER_NAME=${5:-beam_search}
EVAL_N_TOTAL=-1
OUT_PREFIX=config/fold
DB_PICKLE=complete_folds_smiles_holdout_1604314203.pkl
SELECTED_FOLD=$4
#if [[ "$SELECTED_FOLD" == "X" ]]
#then
#	SELECTED_FOLD=
#fi

#if [[ "$EVALUATION_SET" == "casmi" ]]
#then
#	SELECTED_FOLD=0
#fi


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
    if [[ (-f "${DIR}config.yaml") && ( ("$SELECTED_FOLD/" = "$DIR") || ("$SELECTED_FOLD" = "X") ) ]]
    then
        mkdir -p $TDIR
	cp ${DIR}config.yaml ${TDIR}config_edited.yaml
	# remove folders etc. that should be taken from the main config file
	yq e 'del(.base_folder)' -i ${TDIR}config_edited.yaml
	yq e 'del(.eval_folder)' -i ${TDIR}config_edited.yaml
	yq e 'del(.fingerprinter_path)' -i ${TDIR}config_edited.yaml
	yq e 'del(.reinforcement_config)' -i ${TDIR}config_edited.yaml
	yq e 'del(.reinforcement_eval_count)' -i ${TDIR}config_edited.yaml
	yq e 'del(.reinforcement_eval_start)' -i ${TDIR}config_edited.yaml
	yq e 'del(.sirius_bin)' -i ${TDIR}config_edited.yaml
	yq e 'del(.weights_folder)' -i ${TDIR}config_edited.yaml
	yq e 'del(.model_name)' -i ${TDIR}config_edited.yaml
	yq e 'del(.log_folder)' -i ${TDIR}config_edited.yaml
	yq e 'del(.fp_map)' -i ${TDIR}config_edited.yaml
	yq e 'del(.weights)' -i ${TDIR}config_edited.yaml
	yq e '.db_path_eval.fp_map="/msnovelist/data/fingerprint-map/csi_fingerid.csv"' -i ${TDIR}config_edited.yaml
	yq e '.db_path_sample.fp_map="/msnovelist/data/fingerprint-map/csi_fingerid.csv"' -i ${TDIR}config_edited.yaml
	yq e '.db_path_train.fp_map="/msnovelist/data/fingerprint-map/csi_fingerid.csv"' -i ${TDIR}config_edited.yaml
	yq e '.db_path_eval.path="/msnovelist-data/predicted-fingerprints/complete_folds_smiles_holdout_1604314203.pkl"' -i ${TDIR}config_edited.yaml
	yq e '.db_path_sample.path="/msnovelist-data/predicted-fingerprints/complete_folds_smiles_holdout_1604314203.pkl"' -i ${TDIR}config_edited.yaml
	yq e '.db_path_train.path="/msnovelist-data/training-set/combined_0824_v44.db"' -i ${TDIR}config_edited.yaml
	# specifically remove stuff that is going to be re-set
	sed -i.sed '/^eval_detail:/d' ${TDIR}config_edited.yaml
	sed -i.sed '/^evaluation_set:/d' ${TDIR}config_edited.yaml
	sed -i.sed '/^decoder_name:/d' ${TDIR}config_edited.yaml
	sed -i.sed '/^coverage_baseline:/d' ${TDIR}config_edited.yaml
	sed -i.sed '/baseline_sirius_top10_coverage_/d' ${TDIR}config_edited.yaml
	sed -i.sed '/^db_pubchem:/d' ${TDIR}config_edited.yaml
	sed -i.sed '/^eval_counter:/d' ${TDIR}config_edited.yaml
	sed -i.sed '/^eval_id:/d' ${TDIR}config_edited.yaml
	sed -i.sed '/^fingerprinter_cache:/d' ${TDIR}config_edited.yaml
	sed -i.sed '/^eval_n_total:/d' ${TDIR}config_edited.yaml
	sed -i.sed '/^eval_k:/d' ${TDIR}config_edited.yaml
	sed -i.sed '/^eval_kk:/d' ${TDIR}config_edited.yaml
	if [ "$TAG" != "" ]
	then
		sed -i.sed '/^model_tag:/d' ${TDIR}config_edited.yaml
	fi
	sed -i.sed "s|evaluation_v44/complete_folds_smiles.csv|evaluation_v44/$DB_PICKLE|" ${TDIR}config_edited.yaml
	echo eval_detail: True >> ${TDIR}config_edited.yaml
	echo "evaluation_set: '${EVALUATION_SET}'" >> ${TDIR}config_edited.yaml
	echo "decoder_name: '${DECODER_NAME}'" >> ${TDIR}config_edited.yaml
	echo "eval_n_total: ${EVAL_N_TOTAL}" >> ${TDIR}config_edited.yaml
	echo "eval_k: ${EVAL_K}" >> ${TDIR}config_edited.yaml
	echo "eval_kk: ${EVAL_KK}" >> ${TDIR}config_edited.yaml
	if [ "$TAG" != "" ]
	then
		echo "model_tag: ${TAG}" >> ${TDIR}config_edited.yaml  
	fi

	echo "updated or created ${TDIR}config_edited.yaml"
	cat ${TDIR}config_edited.yaml | grep model_tag:
	cat ${TDIR}config_edited.yaml | grep cv_fold:
    fi
done
