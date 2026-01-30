#!/bin/bash
# SLURM submission template for SPINE production (S3DF)
# This is a Jinja2 template - variables in double braces will be substituted

#SBATCH --account={{ account }}
#SBATCH --partition={{ partition }}

#SBATCH --ntasks=1
#SBATCH --gpus={{ gpus }}
#SBATCH --cpus-per-task={{ cpus_per_task }}
#SBATCH --mem-per-cpu={{ mem_per_cpu }}
#SBATCH --time={{ time }}

{% if array_spec %}
#SBATCH --array={{ array_spec }}
{% endif %}


#SBATCH --job-name={{ job_name }}
{% if array_spec %}
#SBATCH --output={{ log_dir }}/{{ job_name }}_%A_%a.out
#SBATCH --error={{ log_dir }}/{{ job_name }}_%A_%a.err
{% else %}
#SBATCH --output={{ log_dir }}/{{ job_name }}_%j.out
#SBATCH --error={{ log_dir }}/{{ job_name }}_%j.err
{% endif %}

{% if dependency %}
#SBATCH --dependency={{ dependency }}
{% endif %}

# Job metadata
echo "=================================="
echo "SPINE Production Job (S3DF)"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
{% if array_spec %}
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
{% endif %}
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

{% if array_spec %}
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
{% else %}
# Single job: just use the file list as is
TASK_FILE_LIST="{{ file_list }}"
FIRST_FILE=$(head -n 1 "$TASK_FILE_LIST")
if [[ ! -f "$FIRST_FILE" ]]; then
    echo "ERROR: Input file not found: $FIRST_FILE"
    exit 1
fi

echo "Processing file list: $TASK_FILE_LIST"
echo "Files in this task:"
cat "$TASK_FILE_LIST"
echo
{% endif %}

# Cap the number Numba threads to 64 (what OpenBLAS was compiled with)
export NUMBA_NUM_THREADS=64
echo "NUMBA_NUM_THREADS set to $NUMBA_NUM_THREADS"

# S3DF: Use Singularity
BIND_PATHS="/sdf/,/fs/"
SINGULARITY_INNER_CMD=""
{% if larcv_basedir %}
SINGULARITY_INNER_CMD="unset which; source {{ larcv_basedir }}/configure.sh; "
{% endif %}
{% if flashmatch %}
SINGULARITY_INNER_CMD="$SINGULARITY_INNER_CMD source $FMATCH_BASEDIR/configure.sh; "
{% endif %}
SINGULARITY_INNER_CMD="$SINGULARITY_INNER_CMD python3 $SPINE_BASEDIR/bin/run.py -S $TASK_FILE_LIST -o {{ output }} -c {{ config }} --log-dir {{ log_dir }}"
SINGULARITY_CMD="singularity exec --bind $BIND_PATHS --nv $CONTAINER_PATH bash -c '$SINGULARITY_INNER_CMD'"
RUN_CMD="$SINGULARITY_CMD"

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
