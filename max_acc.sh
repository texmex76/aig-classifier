#!/bin/bash

max_accuracy=0
max_accuracy_dir=""

# Iterate over all log files in all sub-directories of "results"
for file in $(find results -name '*.log'); do
    # Extract the testing accuracy from the last line
    accuracy=$(tail -n 1 $file | awk '{print $NF}')

    # Update max_accuracy and max_accuracy_dir if the current accuracy is greater
    if (( $(echo "$accuracy > $max_accuracy" | bc -l) )); then
        max_accuracy=$accuracy
        max_accuracy_dir=$(dirname $file)
    fi
done

echo "Maximum Testing Accuracy: $max_accuracy"
echo "Directory with Maximum Testing Accuracy: $max_accuracy_dir"

