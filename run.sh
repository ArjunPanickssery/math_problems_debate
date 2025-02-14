#!/bin/bash

run_with_retry() {
    while true; do
        yes | uv run "$1"
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "Successfully completed $1"
            break
        else
            echo "Error running $1 (exit code: $exit_code). Retrying..."
            # sleep 1  # Optional: add a small delay between retries
        fi
    done
}

# Run scripts sequentially with retry logic
run_with_retry "experiments/init_exp_5.py"
run_with_retry "experiments/init_exp_10.py"