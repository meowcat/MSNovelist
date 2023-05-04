#!/bin/bash

export COMPUTERNAME=DOCKER-LIGHT
export MSNOVELIST_BASE=/msnovelist
export TF_CPP_MIN_LOG_LEVEL=3

# EXPORT_DB_BASE=0

# Get input directory user to adjust all files to that user,
# to avoid rooted files that the user can't delete
USER=`stat -c "%u:%g" /msnovelist-data`

RUNID=`date +%s`
mkdir -p /msnovelist-data/results-$RUNID


cp /msnovelist/config.DOCKER-LIGHT.yaml /msnovelist-data/msnovelist-config-$RUNID.yaml
chown $USER /msnovelist-data/msnovelist-config-$RUNID.yaml
# Write new eval_id into config file
yq e 'del(.eval_id)' -i /msnovelist-data/msnovelist-config-$RUNID.yaml
yq e 'del(.eval_counter)' -i /msnovelist-data/msnovelist-config-$RUNID.yaml
yq e 'del(.eval_folder)' -i /msnovelist-data/msnovelist-config-$RUNID.yaml
# Note: check how yq deals with $, could be done directly above
echo "eval_id: '$RUNID'" >> /msnovelist-data/msnovelist-config-$RUNID.yaml
echo "eval_counter: '0'" >> /msnovelist-data/msnovelist-config-$RUNID.yaml
echo "eval_folder: '/msnovelist-data/'" >> /msnovelist-data/msnovelist-config-$RUNID.yaml

if [[ -f "/msnovelist-data/fingerprint_cache.db" ]]
then
	echo "fingerprint_cache: '/msnovelist-data/fingerprint_cache.db'" >> /msnovelist-data/msnovelist-config-$RUNID.yaml
fi

# Run de novo prediction
cd /msnovelist
python "$MSNOVELIST_BASE/webui/webio.py" -c \
	/msnovelist/data/weights/config.yaml \
	/msnovelist-data/msnovelist-config-$RUNID.yaml

# Set correct ownership for the results
chown -R $USER /msnovelist-data/results-$RUNID
chown -R $USER /msnovelist-data/fingerprint_cache.db





