#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate msnovelist-env
/usr/local/bin/sirius-linux64-headless-4.4.29/bin/sirius $@