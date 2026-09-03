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
#SBATCH --cpus-per-task={{ cpus_per_task }}
#SBATCH --mem={{ mem }}
#SBATCH --time={{ time }}
{% if exclusive %}
#SBATCH --exclusive
{% endif %}

{% if array_spec %}
#SBATCH --array={{ array_spec }}
{% endif %}

#SBATCH --job-name={{ job_name }}
{% if array_spec %}
#SBATCH --output={{ log_dir }}/{{ job_name }}_%A_%a.out
#SBATCH --error={{ log_dir }}/{{ job_name }}_%A_%a.err
{% else %}
#SBATCH --output={{ stdout_path }}
#SBATCH --error={{ stderr_path }}
{% endif %}

{% if dependency %}
#SBATCH --dependency={{ dependency }}
#SBATCH --kill-on-invalid-dep=yes
{% endif %}

# Job metadata
echo "=================================="
echo "{{ command_label|default('SPINE Production Job') }} (NERSC)"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
{% if array_spec %}
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
{% endif %}
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "Config: {{ config }}"
{% if output %}
echo "Output: {{ output }}"
{% elif output_args %}
echo "Output directory: {{ output_dir }}"
echo "Output suffix: {{ output_suffix }}"
{% else %}
echo "Output: defined in config"
{% endif %}
echo "=================================="
echo

# Load environment from configure.sh
if [ -z ${SPINE_PROD_BASEDIR+x} ]; then
    source {{ basedir }}/configure.sh
fi

{% if file_list_pattern and array_spec %}
# Get file list for this array task
TASK_DIR="{{ task_dir_pattern | replace('*', '${SLURM_ARRAY_TASK_ID}') }}"
TASK_FILE_LIST="{{ file_list_pattern | replace('*', '${SLURM_ARRAY_TASK_ID}') }}"

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
{% elif file_list_pattern %}
# Single job: use the single file list
{% if task_dir_pattern %}
TASK_DIR="{{ task_dir_pattern | replace('*', '1') }}"
TASK_FILE_LIST="{{ file_list_pattern | replace('*', '1') }}"
{% else %}
TASK_FILE_LIST="{{ file_list_pattern }}"
{% endif %}
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
{% if input_manifest %}
echo "Using input file list: {{ input_manifest }}"
{% else %}
echo "Using input files defined in the config"
{% endif %}
echo
{% endif %}

# Cap the number Numba threads to 64 (what OpenBLAS was compiled with)
export NUMBA_NUM_THREADS=64
echo "NUMBA_NUM_THREADS set to $NUMBA_NUM_THREADS"

# NERSC: Use Shifter
INNER_CMD=""
{% if larcv_path or flashmatch_path %}
INNER_CMD="unset which; "
{% if larcv_path %}
INNER_CMD="${INNER_CMD}source {{ (larcv_path ~ '/configure.sh')|tojson }}; "
{% endif %}
{% if flashmatch_path %}
INNER_CMD="${INNER_CMD}source {{ (flashmatch_path ~ '/configure.sh')|tojson }}; "
{% endif %}
{% endif %}
{% if custom_command %}
INNER_CMD="${INNER_CMD}{{ custom_command }}"
{% else %}
INNER_CMD="${INNER_CMD}{{ spine_cmd|default('spine') }}{% if file_list_pattern %} -S $TASK_FILE_LIST{% endif %} {{ output_args }} -c {{ config }} --log-dir {{ spine_log_dir|default(log_dir) }}{% if spine_cli_overrides %} {{ spine_cli_overrides }}{% endif %}"
{% endif %}

# Add Shifter modules requested by the job profile/options
SHIFTER_MODULES=()
if [[ {{ gpus_per_node|default(0) }} -gt 0 ]]; then
    SHIFTER_MODULES+=("--module gpu")
fi
{% if cvmfs %}
SHIFTER_MODULES+=("--module=cvmfs")
{% endif %}

RUN_CMD="srun -n 1 shifter ${SHIFTER_MODULES[*]} --image=$SPINE_CONTAINER_TAG bash -c \"$INNER_CMD\""

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
