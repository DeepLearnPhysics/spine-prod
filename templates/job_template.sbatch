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

# Detect site and configure container execution
if [[ -d "/sdf" ]]; then
    # S3DF - uses Singularity with bind mounts
    SITE="s3df"
    BIND_PATHS="/sdf/,/fs/"
elif [[ -d "/global/cfs" ]]; then
    # NERSC - uses Shifter with srun
    SITE="nersc"
    BIND_PATHS="/global/cfs/,/pscratch/sd/"
else
    # Default/unknown site
    SITE="unknown"
    BIND_PATHS="$PWD"
fi

# Build execution command based on site
if [[ "$SITE" == "nersc" ]]; then
    # NERSC: Use srun with shifter
    # Build the command to run inside container
    INNER_CMD=""
    {% if larcv_basedir %}
    INNER_CMD="unset which; source {{ larcv_basedir }}/configure.sh; "
    {% endif %}
    {% if flashmatch %}
    INNER_CMD="${INNER_CMD}source \$FMATCH_BASEDIR/configure.sh; "
    {% endif %}
    INNER_CMD="${INNER_CMD}python3 \$SPINE_BASEDIR/bin/run.py -S $TASK_FILE_LIST -o {{ output }} -c {{ config }}"

    # Execute with srun and shifter
    RUN_CMD="srun -n 1 shifter --image=$CONTAINER_TAG \"$INNER_CMD\""
else
    # S3DF or other: Use Singularity
    SINGULARITY_CMD="singularity exec --bind $BIND_PATHS --nv $CONTAINER_PATH bash -c \""
    {% if larcv_basedir %}
    SINGULARITY_CMD="$${SINGULARITY_CMD}unset which; source {{ larcv_basedir }}/configure.sh; "
    {% endif %}
    {% if flashmatch %}
    SINGULARITY_CMD="$${SINGULARITY_CMD}source $$FMATCH_BASEDIR/configure.sh; "
    {% endif %}
    SINGULARITY_CMD="$${SINGULARITY_CMD}python3 $$SPINE_BASEDIR/bin/run.py -S $$TASK_FILE_LIST -o {{ output }} -c {{ config }}\""
    RUN_CMD="$SINGULARITY_CMD"
fi

# Execute
echo "Site: $SITE"
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
