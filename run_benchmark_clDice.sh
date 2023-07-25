#!/bin/sh

# Stop on error
set -e
# Stop on unset variable
set -u

if [ "$#" -lt 4 ]; then
    echo "Run benchmarks on a the given binary volume (DATASET) for each of the given scales (SCALES)"
    echo "BENCHMARK_TYPE is either mirror or scale (see paper)."
    echo "RUN_TYPE is one of 'bench', 'perf', 'debug', 'run', 'memcheck', and 'heaptrack'."
    echo "Results will be saved in the 'runs' subfolder of the selected BENCHMARK_TYPE."
    echo "usage: run_benchmark.sh RUN_TYPE BENCHMARK_TYPE DATASET [SCALES ...]"
    exit 1
fi

script_dir=$(dirname $0)
config="$script_dir/../config.sh"
. $config

run_type="$1"
shift
benchmark="$1"
shift
dataset="$1"
shift
scales="$@"
echo Running benchmark $benchmark for dataset $dataset with scales "[" $scales "]"

workspace="$benchmark/workspace.vws"
script="$benchmark/script.py"

revision=$(${VOREEN_TOOL_PATH} -platform minimal --revision)
if [ -z ${revision} ]; then
    echo "Failed to find revision"
    exit 1
fi
result_log_dir="$benchmark/runs/${revision}"
mkdir -p ${result_log_dir}
result_log="${result_log_dir}/run_$(date +%Y-%m-%d-%H:%M:%S).log"
echo "Using log file ${result_log}"

# Clean up junk/previous output
ls ./*.vvg &> /dev/null && rm -f ./*.vvg
ls ./*.vvg.gz &> /dev/null && rm -f ./*.vvg.gz

# Run benchmark
for scale in $scales; do
    commandline="${VOREEN_TOOL_PATH} -platform minimal --workspace ${workspace} --script ${script} --scriptArgs \"$dataset\" $scale --tempdir ${TMP_DIR_PATH}"
    echo "Running command: ${commandline}"
    case "$run_type" in
        "bench")
            time -v $commandline 2>&1 | tee -a ${result_log}
            ;;
        "perf")
            perf record --call-graph lbr -- $commandline
            ;;
        "memcheck")
            valgrind --tool=memcheck --log-file=vg.log -- $commandline
            ;;
        "heaptrack")
            heaptrack $commandline
            ;;
        "debug")
            echo ugdb -- $commandline
            ugdb -- $commandline
            ;;
        "run")
            $commandline
            ;;
        *) echo "Run type must be either 'bench', 'perf', 'debug', 'run', 'memcheck' or 'heaptrack'"; exit 1 ;;
    esac

done
