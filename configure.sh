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
    export SPINE_CONFIG_PATH=$SPINE_PROD_BASEDIR/infer:$SPINE_PROD_BASEDIR/train
else
    export SPINE_CONFIG_PATH=$SPINE_PROD_BASEDIR/infer:$SPINE_PROD_BASEDIR/train:$SPINE_CONFIG_PATH
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

# Define path to OpT0Finder
#export FMATCH_BASEDIR=/sdf/data/neutrino/software/OpT0Finder_legacy
export FMATCH_BASEDIR=/sdf/data/neutrino/software/OpT0Finder

# Define path to the singularity container
#export SINGULARITY_PATH=/sdf/group/neutrino/images/larcv2_ub20.04-cuda11.6-pytorch1.13-larndsim.sif
#export SINGULARITY_PATH=/sdf/group/neutrino/images/larcv2_ub22.04-cuda12.1-pytorch2.4.0-larndsim-2024-09-03.sif
export SINGULARITY_PATH=/sdf/group/neutrino/images/larcv2_ub2204-cuda121-torch251-larndsim.sif

echo
printf "\033[93mSPINE_PROD\033[00m FYI shell env. may useful for external packages:\n"
printf "    \033[95mSPINE_PROD_BASEDIR\033[00m = $SPINE_PROD_BASEDIR\n"
printf "    \033[95mSPINE_CONFIG_PATH\033[00m  = $SPINE_CONFIG_PATH\n"
printf "    \033[95mSPINE_BASEDIR\033[00m      = $SPINE_BASEDIR\n"
printf "    \033[95mFMATCH_BASEDIR\033[00m     = $FMATCH_BASEDIR\n"
printf "    \033[95mSINGULARITY_PATH\033[00m   = $SINGULARITY_PATH\n"

echo
echo "Finished configuration."
echo
