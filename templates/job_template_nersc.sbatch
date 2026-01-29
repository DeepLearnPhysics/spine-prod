#!/bin/bash
# SLURM submission template for SPINE production (NERSC)
# This is a Jinja2 template - variables in double braces will be substituted

#SBATCH --account={{ account }}
#SBATCH --qos={{ qos }}
{% if constraint %}
#SBATCH --constraint={{ constraint }}
{% endif %}

#SBATCH --nodes=1
#SBATCH --gpus-per-node={{ gpus_per_node }}
#SBATCH --cpus-per-node={{ cpus_per_node }}
#SBATCH --mem={{ mem }}
#SBATCH --time={{ time }}

#SBATCH --array={{ array_spec }}

#SBATCH --job-name={{ job_name }}
#SBATCH --output={{ log_dir }}/{{ job_name }}_%A_%a.out
#SBATCH --error={{ log_dir }}/{{ job_name }}_%A_%a.err

{% if dependency %}
#SBATCH --dependency={{ dependency }}
{% endif %}

# Job metadata
echo "=================================="
echo "SPINE Production Job (NERSC)"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "Config: {{ config }}"
echo "Output: {{ output }}"
echo "=================================="
echo

# Load environment from configure.sh
if [ -z $${SPINE_PROD_BASEDIR+x} ]; then
    source {{ basedir }}/configure.sh
fi

# Get file list for this array task
FILE_GROUP=$(sed -n "${SLURM_ARRAY_TASK_ID}p" {{ file_list }})

# Create a temporary file list for this task
TASK_FILE_LIST="{{ log_dir }}/task_files_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
echo "$FILE_GROUP" | tr ',' '\n' > "$TASK_FILE_LIST"

# Verify first file exists
FIRST_FILE=$(head -n 1 "$TASK_FILE_LIST")
if [[ ! -f "$FIRST_FILE" ]]; then
    echo "ERROR: Input file not found: $FIRST_FILE"
    exit 1
fi

echo "Processing file list: $TASK_FILE_LIST"
echo "Files in this task:"
cat "$TASK_FILE_LIST"
echo

# Cap the number Numba threads to 64 (what OpenBLAS was compiled with)
export NUMBA_NUM_THREADS=64
echo "NUMBA_NUM_THREADS set to $NUMBA_NUM_THREADS"

# NERSC: Use Shifter
INNER_CMD=""
{% if larcv_basedir %}
INNER_CMD="unset which; source {{ larcv_basedir }}/configure.sh; "
{% endif %}
{% if flashmatch %}
INNER_CMD="${INNER_CMD}source $FMATCH_BASEDIR/configure.sh; "
{% endif %}
INNER_CMD="${INNER_CMD}python3 $SPINE_BASEDIR/bin/run.py -S $TASK_FILE_LIST -o {{ output }} -c {{ config }} --log-dir {{ log_dir }}"

# Add --module gpu if using a GPU profile
SHIFTER_MODULE=""
if [[ {{ gpus|default(0) }} -gt 0 ]]; then
    SHIFTER_MODULE="--module gpu"
fi

RUN_CMD="srun -n 1 shifter $SHIFTER_MODULE --image=$CONTAINER_TAG \"$INNER_CMD\""

# Execute
echo "Running: $RUN_CMD"
echo
eval $RUN_CMD

# Check exit status
EXIT_STATUS=$?
echo
echo "=================================="
echo "Finished: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "=================================="

exit $EXIT_STATUS
