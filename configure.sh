#!/bin/bash

# Clean up previously set env, register this one
if [[ -z $FORCE_SPINE_PROD_BASEDIR ]]; then
    where="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    export SPINE_PROD_BASEDIR=${where}
else
    export SPINE_PROD_BASEDIR=$FORCE_SPINE_PROD_BASEDIR
fi

# Setup config search path (prepend once to any existing paths)
SPINE_PROD_CONFIG_PATH=$SPINE_PROD_BASEDIR/config
if [[ -z $SPINE_CONFIG_PATH ]]; then
    export SPINE_CONFIG_PATH=$SPINE_PROD_CONFIG_PATH
elif [[ :$SPINE_CONFIG_PATH: != *:$SPINE_PROD_CONFIG_PATH:* ]]; then
    export SPINE_CONFIG_PATH=$SPINE_PROD_CONFIG_PATH:$SPINE_CONFIG_PATH
fi

# Define the SPINE container release used by production jobs.
if [[ -z $SPINE_CONTAINER_VERSION ]]; then
    export SPINE_CONTAINER_VERSION=0.12.1
fi

# Define path to the container (Singularity/Apptainer .sif file)
SPINE_CONTAINER_PATH_VERSION=${SPINE_CONTAINER_VERSION//./-}
DEFAULT_CONTAINER_PATH=/sdf/data/neutrino/images/spine_v${SPINE_CONTAINER_PATH_VERSION}.sif
if [[ -z $SPINE_CONTAINER_PATH || ${SPINE_CONTAINER_PATH_AUTO:-0} == 1 ]]; then
    export SPINE_CONTAINER_PATH=$DEFAULT_CONTAINER_PATH
    export SPINE_CONTAINER_PATH_AUTO=1
else
    export SPINE_CONTAINER_PATH_AUTO=0
fi

# Define container tag (Shifter image tag for NERSC or local docker execution)
DEFAULT_CONTAINER_TAG=docker:ghcr.io/deeplearnphysics/spine:${SPINE_CONTAINER_VERSION}
if [[ -z $SPINE_CONTAINER_TAG ]]; then
    export SPINE_CONTAINER_TAG=$DEFAULT_CONTAINER_TAG
fi

# If ICARUS_DATA_DIR is not set, default to the standard location on CVMFS.
if [[ -z $ICARUS_DATA_DIR ]]; then
    export ICARUS_DATA_DIR=/cvmfs/icarus.opensciencegrid.org/products/icarus/icarus_data
fi

echo
printf "\033[93mSPINE_PROD\033[00m FYI shell env. may useful for external packages:\n"
printf "    \033[95mSPINE_PROD_BASEDIR\033[00m      = $SPINE_PROD_BASEDIR\n"
printf "    \033[95mSPINE_CONFIG_PATH\033[00m       = $SPINE_CONFIG_PATH\n"
printf "    \033[95mSPINE_CONTAINER_VERSION\033[00m = $SPINE_CONTAINER_VERSION\n"
printf "    \033[95mSPINE_CONTAINER_PATH\033[00m    = $SPINE_CONTAINER_PATH\n"
printf "    \033[95mSPINE_CONTAINER_TAG\033[00m     = $SPINE_CONTAINER_TAG\n"
printf "    \033[95mICARUS_DATA_DIR\033[00m         = $ICARUS_DATA_DIR\n"

echo
echo "Finished configuration."
echo
