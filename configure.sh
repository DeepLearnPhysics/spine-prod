#!/bin/bash

# Clean up previously set env, register this one
if [[ -z $FORCE_SPINE_PROD_BASEDIR ]]; then
    where="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    export SPINE_PROD_BASEDIR=${where}
else
    export SPINE_PROD_BASEDIR=$FORCE_SPINE_PROD_BASEDIR
fi

# Setup config search path (prepend to any existing paths)
if [[ -z $SPINE_CONFIG_PATH ]]; then
    export SPINE_CONFIG_PATH=$SPINE_PROD_BASEDIR/config
else
    export SPINE_CONFIG_PATH=$SPINE_PROD_BASEDIR/config:$SPINE_CONFIG_PATH
fi

# Define path to SPINE
# Default to submodule if it exists, otherwise fall back to system installation
if [[ -z $SPINE_BASEDIR ]]; then
    if [[ -d "$SPINE_PROD_BASEDIR/spine" ]]; then
        export SPINE_BASEDIR=$SPINE_PROD_BASEDIR/spine
    else
        export SPINE_BASEDIR=/sdf/data/neutrino/software/spine
    fi
fi

# If FMATCH_BASEDIR is not set, default to the standard location on S3DF.
if [[ -z $FMATCH_BASEDIR ]]; then
    export FMATCH_BASEDIR=/sdf/data/neutrino/software/OpT0Finder
fi

# If ICARUS_DATA_DIR is not set, default to the standard location on CVMFS.
if [[ -z $ICARUS_DATA_DIR ]]; then
    export ICARUS_DATA_DIR=/cvmfs/icarus.opensciencegrid.org/products/icarus/icarus_data
fi

# Define path to the container (Singularity/Apptainer .sif file)
if [[ -z $CONTAINER_PATH ]]; then
    export CONTAINER_PATH=/sdf/group/neutrino/images/larcv2_ub2204-cuda121-torch251-larndsim.sif
fi

# Define container tag (Shifter image tag for NERSC)
export CONTAINER_TAG=deeplearnphysics/larcv2:ub2204-cu121-torch251-larndsim

echo
printf "\033[93mSPINE_PROD\033[00m FYI shell env. may useful for external packages:\n"
printf "    \033[95mSPINE_PROD_BASEDIR\033[00m = $SPINE_PROD_BASEDIR\n"
printf "    \033[95mSPINE_CONFIG_PATH\033[00m  = $SPINE_CONFIG_PATH\n"
printf "    \033[95mSPINE_BASEDIR\033[00m      = $SPINE_BASEDIR\n"
printf "    \033[95mFMATCH_BASEDIR\033[00m     = $FMATCH_BASEDIR\n"
printf "    \033[95mICARUS_DATA_DIR\033[00m    = $ICARUS_DATA_DIR\n"
printf "    \033[95mCONTAINER_PATH\033[00m     = $CONTAINER_PATH\n"
printf "    \033[95mCONTAINER_TAG\033[00m      = $CONTAINER_TAG\n"

echo
echo "Finished configuration."
echo
