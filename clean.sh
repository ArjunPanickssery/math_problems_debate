#!/bin/bash

# Function to delete or keep n most recent items in a directory
cleanup() {
    local dir=$1
    local keep=$2

    if [ -d "$dir" ]; then
        if [ "$keep" -gt 0 ]; then
            # Get a list of items in the directory sorted by modification time, excluding the latest $keep items
            to_delete=($(ls -1t "$dir" | tail -n +$((keep + 1))))
            for item in "${to_delete[@]}"; do
                rm -rf "$dir/$item"
            done
        else
            # Remove everything in the directory if keep is 0 or not specified
            rm -rf "$dir"/*
        fi
        echo "Cleaned up $dir, kept $keep most recent items"
    else
        echo "Directory $dir does not exist"
    fi
}

# Default value for keeping items is 0 (delete all)
keep=0

# Parse arguments
for arg in "$@"; do
    case $arg in
        --cache)
            cleanup ".cache" "$keep"
            ;;
        --costly)
            cleanup ".costly" "$keep"
            ;;
        --logs)
            cleanup ".logs" "$keep"
            ;;
        --tests)
            cleanup "tests/test_results" "$keep"
            ;;
        --all)
            cleanup ".cache" "$keep"
            cleanup ".costly" "$keep"
            cleanup ".logs" "$keep"
            cleanup "tests/test_results" "$keep"
            ;;
        -n=*|--keep=*)
            keep="${arg#*=}"
            ;;
        *)
            echo "Unknown option: $arg"
            ;;
    esac
done


# rm -r .cache/
# rm -r .costly/
# rm -r .logs/
# rm -r tests/test_results/
