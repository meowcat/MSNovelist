#!/bin/bash

export COMPUTERNAME=DOCKER-LIGHT
export MSNOVELIST_BASE=/msnovelist
export TF_CPP_MIN_LOG_LEVEL=3


if [[ "$NAMING" != "" ]]
then
	NAMING_ARG="--naming-convention=$NAMING"
fi

SIRIUS_SETTINGS=${2:-"${NAMING_ARG} formula -p qtof structure -d ALL_BUT_INSILICO"}
#SIRIUS_SETTINGS=${2:-"formula -p qtof structure"}
# EXPORT_DB_BASE=0

eval "$(conda shell.bash hook)"
conda activate msnovelist-env

# Get input directory user to adjust all files to that user,
# to avoid rooted files that the user can't delete
USER=`stat -c "%u:%g" /msnovelist-data`

RUNID=`date +%s`
mkdir -p /msnovelist-data/results-$RUNID


cp /msnovelist/config.DOCKER-LIGHT.yaml /msnovelist-data/msnovelist-config-$RUNID.yaml
chown $USER /msnovelist-data/msnovelist-config-$RUNID.yaml


# If this is a file with spectra: process with SIRIUS and use resulting path as input path for MSNovelist
if [[ ! -d "/msnovelist-data/$1" ]]
then
	HOME=/tmp sirius.sh --log=WARNING -i "/msnovelist-data/$1" -o "/msnovelist-data/sirius-$RUNID" $SIRIUS_SETTINGS
	chown -R $USER /msnovelist-data/sirius-$RUNID
	echo "sirius_project_input: /msnovelist-data/sirius-$RUNID" >> /msnovelist-data/msnovelist-config-$RUNID.yaml
# Otherwise use input directory as input path for MSNovelist
else
	echo "sirius_project_input: /msnovelist-data/$1" >> /msnovelist-data/msnovelist-config-$RUNID.yaml
	chown -R $USER /msnovelist-data/$1
fi

# Write new eval_id into config file
yq e 'del(.eval_id)' -i /msnovelist-data/msnovelist-config-$RUNID.yaml
yq e 'del(.eval_counter)' -i /msnovelist-data/msnovelist-config-$RUNID.yaml
yq e 'del(.eval_folder)' -i /msnovelist-data/msnovelist-config-$RUNID.yaml
# Note: check how yq deals with $, could be done directly above
echo "eval_id: '$RUNID'" >> /msnovelist-data/msnovelist-config-$RUNID.yaml
echo "eval_counter: '0'" >> /msnovelist-data/msnovelist-config-$RUNID.yaml
echo "eval_folder: '/msnovelist-data/results-$RUNID/'" >> /msnovelist-data/msnovelist-config-$RUNID.yaml


if [[ -f "/msnovelist-data/fingerprint_cache.db" ]]
then
	echo "fingerprint_cache: '/msnovelist-data/fingerprint_cache.db'" >> /msnovelist-data/msnovelist-config-$RUNID.yaml
fi

# Run de novo prediction
cd /msnovelist
python "$MSNOVELIST_BASE/decode_applied.py" -c \
	/msnovelist/data/weights/config.yaml \
	/msnovelist-data/msnovelist-config-$RUNID.yaml

# If required, also export database results for comparison (set EXPORT_DB_BASE=1)
if [[ "$EXPORT_DB_BASE" == "1" ]]
then
	python "$MSNOVELIST_BASE/evaluation/applied_sirius_comparison.py" -c \
		/msnovelist/data/weights/config.yaml \
		/msnovelist-data/msnovelist-config-$RUNID.yaml
fi

# Set correct ownership for the results
chown -R $USER /msnovelist-data/results-$RUNID
chown -R $USER /msnovelist-data/fingerprint_cache.db





