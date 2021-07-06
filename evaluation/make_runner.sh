#!/bin/bash

MODELDIR=$1
SELECTED_FOLD=$3
PREFIX_MODEL=config/fold
PREFIX_WEIGHTS=config/weights_fold
TAG=$2

DIRS=`ls $MODELDIR | grep -E '[0-9]+'`

echo "#!/bin/bash"
echo ""

for DIR in ${DIRS} 
do
	if [[ ( "$SELECTED_FOLD" == "$DIR" ) || ( "$SELECTED_FOLD" == "X" ) ]]
	then
		echo "TRAINED=$1/$PREFIX_MODEL$DIR WEIGHTS=$1/$PREFIX_WEIGHTS$DIR TAG=$TAG bsub < ${MSNOVELIST_ROOT_SLASH}evaluation.sh"
	fi
done


