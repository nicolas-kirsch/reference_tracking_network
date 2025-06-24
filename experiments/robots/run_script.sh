#!/bin/bash

# Ensure the script stops if any command fails
set -e


for alpha_formation in  100 250 500; do
    echo "Running simulation with alpha_formation: $alpha_formation"
    python run.py --alpha-formation $alpha_formation --num-rollouts 600
    echo "Finished running with alpha_formation: $alpha_formation"
    echo "------------------------------------"
done
