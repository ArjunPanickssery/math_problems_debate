#!/bin/bash

# Function to remove all items except the most recent n items
cleanup_folder() {
    folder=$1
    keep=$2
    
    # Check if the folder exists
    if [ -d "$folder" ]; then
        # List items sorted by modification time, skipping the most recent $keep items
        items_to_remove=$(ls -t "$folder" | tail -n +$((keep + 1)))
        for item in $items_to_remove; do
            rm -r "$folder/$item"
        done
    else
        echo "Folder $folder does not exist."
    fi
}

# Default keep value
keep=0

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cache)
            action_cache=true
            if [[ "$2" =~ ^[0-9]+$ ]]; then
                keep=$2
                shift
            fi
            ;;
        --costly)
            action_costly=true
            if [[ "$2" =~ ^[0-9]+$ ]]; then
                keep=$((3 * $2)) # Set keep to 3 * the provided value for --costly
                shift
            fi
            ;;
        --logs)
            action_logs=true
            if [[ "$2" =~ ^[0-9]+$ ]]; then
                keep=$2
                shift
            fi
            ;;
        --tests)
            action_tests=true
            if [[ "$2" =~ ^[0-9]+$ ]]; then
                keep=$2
                shift
            fi
            ;;
        --all)
            action_cache=true
            action_costly=true
            action_logs=true
            action_tests=true
            if [[ "$2" =~ ^[0-9]+$ ]]; then
                keep=$2
                shift
            fi
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

# Run actions based on the parsed arguments
if [[ "$action_cache" == true ]]; then
    echo "Cleaning up .cache/ folder, keeping $keep most recent items..."
    cleanup_folder ".cache" $keep
fi

if [[ "$action_costly" == true ]]; then
    echo "Cleaning up .costly/ folder, keeping $keep most recent items..."
    cleanup_folder ".costly" $keep
fi

if [[ "$action_logs" == true ]]; then
    echo "Cleaning up .logs/ folder, keeping $keep most recent items..."
    cleanup_folder ".logs" $keep
fi

if [[ "$action_tests" == true ]]; then
    echo "Cleaning up tests/test_results/ folder, keeping $keep most recent items..."
    cleanup_folder "tests/test_results" $keep
fi
