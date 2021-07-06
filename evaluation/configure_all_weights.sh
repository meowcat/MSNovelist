#!/bin/bash

WEIGHTS_ROOT=/msnovelist-data/weights
MSNOVELIST_ROOT=/msnovelist
export MSNOVELIST_ROOT_SLASH=$MSNOVELIST_ROOT/
WEIGHTS_SOURCE=weights_final
WEIGHTS_TARGET=weights_models
RUNNER_TARGET=$MSNOVELIST_ROOT/evaluation_runners

mkdir -p $WEIGHTS_ROOT/$WEIGHTS_TARGET
mkdir -p $RUNNER_TARGET
rm -rf $WEIGHTS_ROOT/$WEIGHTS_TARGET/*


MODELS_LIST=`ls $WEIGHTS_ROOT/$WEIGHTS_SOURCE`

# Arguments: 
# $1 is the target name
# $2 is the dataset
# $3 is top-k number
# $4 is the selected fold (empty: all folds)
make_weights_one_target () {
	TARGETNAME=$1
	DATASET=$2
	TOPK=$3
	SELECTED_FOLD=$4
	cp -r $WEIGHTS_ROOT/$WEIGHTS_SOURCE $WEIGHTS_ROOT/$WEIGHTS_TARGET/$TARGETNAME
	cd $WEIGHTS_ROOT/$WEIGHTS_TARGET/$TARGETNAME
	for MODEL in $MODELS_LIST
	do
		echo $TARGETNAME - dataset $DATASET, top $TOPK: $MODEL
		cd $WEIGHTS_ROOT/$WEIGHTS_TARGET/$TARGETNAME/$MODEL
		echo `pwd`
		$MSNOVELIST_ROOT/evaluation/make_config.sh
		$MSNOVELIST_ROOT/evaluation/convert_config_files.sh $TOPK $TARGETNAME-$MODEL $DATASET $SELECTED_FOLD
		$MSNOVELIST_ROOT/evaluation/make_runner.sh $WEIGHTS_ROOT/$WEIGHTS_TARGET/$TARGETNAME/$MODEL $TARGETNAME-$MODEL $SELECTED_FOLD > $RUNNER_TARGET/runner-$TARGETNAME-$MODEL.sh
		chmod 755 $RUNNER_TARGET/runner-$TARGETNAME-$MODEL.sh
	done
}

# This one specifically goes for the pure-LSTM and adds a stochastic configuration (DECODER_NAME=stochastic_sampler)
make_stochastic_weights_one_target () {
	TARGETNAME=$1
	DATASET=$2
	TOPK=$3
	SELECTED_FOLD=$4
	CHOSEN_MODEL="G-selected"
	# MODEL is the output model name so the other commands stay as they are above
	MODEL=G-selected-stochastic
	cp -r $WEIGHTS_ROOT/$WEIGHTS_SOURCE/$CHOSEN_MODEL $WEIGHTS_ROOT/$WEIGHTS_TARGET/$TARGETNAME/$MODEL
	echo $TARGETNAME - dataset $DATASET, top $TOPK: $MODEL
	cd $WEIGHTS_ROOT/$WEIGHTS_TARGET/$TARGETNAME/$MODEL
	echo `pwd`
	$MSNOVELIST_ROOT/evaluation/make_config.sh
	$MSNOVELIST_ROOT/evaluation/convert_config_files.sh $TOPK $TARGETNAME-$MODEL $DATASET $SELECTED_FOLD stochastic_sampler
	$MSNOVELIST_ROOT/evaluation/make_runner.sh $WEIGHTS_ROOT/$WEIGHTS_TARGET/$TARGETNAME/$MODEL $TARGETNAME-$MODEL $SELECTED_FOLD > $RUNNER_TARGET/runner-$TARGETNAME-$MODEL.sh
	chmod 755 $RUNNER_TARGET/runner-$TARGETNAME-$MODEL.sh
}



make_weights_one_target sirius-top128 sirius 128 X 
make_weights_one_target sirius-top16 sirius 16 X
make_weights_one_target casmi-top128 casmi 128 0
make_weights_one_target casmi-top16 casmi 16 0

make_stochastic_weights_one_target sirius-top128 sirius 128 X
make_stochastic_weights_one_target sirius-top16 sirius 16 X
make_stochastic_weights_one_target casmi-top128 casmi 128 0
make_stochastic_weights_one_target casmi-top16 casmi 16 0


