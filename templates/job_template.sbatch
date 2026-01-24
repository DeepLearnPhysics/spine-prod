#!/bin/bash
# SLURM submission template for SPINE production
# This is a Jinja2 template - variables in double braces will be substituted

#SBATCH --account={{ account }}
#SBATCH --partition={{ partition }}
{% if qos %}
#SBATCH --qos={{ qos }}
{% endif %}

#SBATCH --ntasks=1
#SBATCH --cpus-per-task={{ cpus_per_task }}
#SBATCH --mem-per-cpu={{ mem_per_cpu }}
#SBATCH --time={{ time }}
#SBATCH --gpus={{ gpus }}
{% if constraint %}
#SBATCH --constraint={{ constraint }}
{% endif %}

#SBATCH --array={{ array_spec }}

#SBATCH --job-name={{ job_name }}
#SBATCH --output={{ log_dir }}/{{ job_name }}_%A_%a.out
#SBATCH --error={{ log_dir }}/{{ job_name }}_%A_%a.err
{% if dependency %}
#SBATCH --dependency={{ dependency }}
{% endif %}

# Job metadata
echo "=================================="
echo "SPINE Production Job"
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
# Each line in the chunk file contains comma-separated files for one task
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

# Detect site and set bind paths
if [[ -d "/sdf" ]]; then
    # S3DF
    BIND_PATHS="/sdf/,/fs/"
elif [[ -d "/global/cfs" ]]; then
    # NERSC
    BIND_PATHS="/global/cfs/,/pscratch/sd/"
else
    # Default/unknown site - use minimal binding
    BIND_PATHS="$PWD"
fi

# Build singularity command
SINGULARITY_CMD="singularity exec --bind $BIND_PATHS --nv $SINGULARITY_PATH bash -c \""

# Add LArCV environment if specified
{% if larcv_basedir %}
SINGULARITY_CMD="$${SINGULARITY_CMD}unset which; source {{ larcv_basedir }}/configure.sh; "
{% endif %}

# Add flashmatch environment if needed
{% if flashmatch %}
SINGULARITY_CMD="$${SINGULARITY_CMD}source $$FMATCH_BASEDIR/configure.sh; "
{% endif %}

# Add SPINE execution command (using -S for source list)
SINGULARITY_CMD="$${SINGULARITY_CMD}python3 $$SPINE_BASEDIR/bin/run.py -S $$TASK_FILE_LIST -o {{ output }} -c {{ config }}\""

# Execute
echo "Running: $SINGULARITY_CMD"
echo
eval $SINGULARITY_CMD

# Check exit status
EXIT_STATUS=$?
echo
echo "=================================="
echo "Finished: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "=================================="

exit $EXIT_STATUS
