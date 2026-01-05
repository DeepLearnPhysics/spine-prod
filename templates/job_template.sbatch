#!/bin/bash
# SLURM submission template for SPINE production
# This is a Jinja2 template - variables in double braces will be substituted

#SBATCH --account={{ account }}
#SBATCH --partition={{ partition }}

#SBATCH --ntasks=1
#SBATCH --cpus-per-task={{ cpus_per_task }}
#SBATCH --mem-per-cpu={{ mem_per_cpu }}
#SBATCH --time={{ time }}
#SBATCH --gpus={{ gpus }}

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

# Get input file(s) for this array task
FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" {{ file_list }})

# Verify file exists
if [[ ! -f $${FILE%%,*} ]]; then
    echo "ERROR: Input file not found: $${FILE%%,*}"
    exit 1
fi

echo "Processing file(s): $FILE"
echo

# Build singularity command
SINGULARITY_CMD="singularity exec --bind /sdf/,/fs/ --nv $SINGULARITY_PATH bash -c \""

# Add LArCV environment if specified
{% if larcv_basedir %}
SINGULARITY_CMD="$${SINGULARITY_CMD}unset which; source {{ larcv_basedir }}/configure.sh; "
{% endif %}

# Add flashmatch environment if needed
{% if flashmatch %}
SINGULARITY_CMD="$${SINGULARITY_CMD}source $$FMATCH_BASEDIR/configure.sh; "
{% endif %}

# Add SPINE execution command
SINGULARITY_CMD="$${SINGULARITY_CMD}python3 $$SPINE_BASEDIR/bin/run.py -s $$FILE -o {{ output }} -c {{ config }}\""

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
