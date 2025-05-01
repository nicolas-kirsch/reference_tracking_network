#!/bin/bash

# Ensure the script stops if any command fails
set -e

# # sensitivity analysis epochs
# for epochs in 100 200 300 400 500 600 700 800 900 1000; do
#     echo "Running simulation with epochs: $epochs"
#     python run.py --epochs "$epochs" --record epochs --filename epochs7
#     echo "Finished running with epochs: $epochs"
#     echo "------------------------------------"
# done

# sensitivity analysis for epochs and num-rollouts
# for epochs in 600 800 1000; do
#     for num_rollouts in 600 800 1000; do
#         echo "Running simulation with epochs: $epochs and num-rollouts: $num_rollouts"
#         python run.py \
#             --epochs "$epochs" \
#             --num-rollouts "$num_rollouts" \
#             --record epochs_rollouts \
#             --filename epochs_rollouts7 \
#             --x-interval "(-1, 5)" \
#             --y-interval "(4, 4.1)" \
#             --save-path "saved_results_v8"
#         echo "Finished running with epochs: $epochs and num-rollouts: $num_rollouts"
#         echo "------------------------------------"
#     done
# done


# # # # Sensitivity analysis for dim-internal and dim-nl
# for dim_internal in 4 8 12 16; do
#     for dim_nl in 4 8 12 16; do
#         echo "Running simulation with dim-internal: $dim_internal and dim-nl: $dim_nl"
#         python run.py --dim-internal "$dim_internal" --dim-nl "$dim_nl" --record dim --filename dim10 --save-path "saved_results_v10"
#         echo "Finished running with dim-internal: $dim_internal and dim-nl: $dim_nl"
#         echo "------------------------------------"
#     done
# done

# # # # Sensitivity analysis for num-rollouts
# for num_rollouts in 10 50 100 200 300 400 500 600 700 800; do
#     echo "Running simulation with num-rollouts: $num_rollouts"
#     python run.py --num-rollouts "$num_rollouts" --record rollouts --filename rollouts10 --save-path "saved_results_v10" 
#     echo "Finished running with num-rollouts: $num_rollouts"
#     echo "------------------------------------"
# done

# Sensitivity analysis for rt-epochs
# for rt_epoch in 0.0 0.2 0.4 0.6 0.8 1; do
#     echo "Running simulation with rt-epochs: $rt_epoch"
#     python run_rt_v3.py --epochs 400 --rt-epochs "$rt_epoch" --record rollouts --filename rt_epochs15 --save-path "saved_results_v15"  --use-previous-params True
#     echo "Finished running with rt-epochs: $rt_epoch"
#     echo "------------------------------------"
# done
for rt_epoch in 0.0 0.2 0.4 0.6 0.8 1; do
    echo "Running simulation with rt-epochs: $rt_epoch"
    python run_rt_v3.py --epochs 400 --rt-epochs "$rt_epoch" --record rollouts --filename rt_epochs16 --save-path "saved_results_v16"  --use-previous-params False --model-path "saved_models/trained_controller_0.pt"
    echo "Finished running with rt-epochs: $rt_epoch"
    echo "------------------------------------"
done
echo "All sensitivity analysis simulations completed."


# python run.py --obstacle-shape 6 --dim-internal 2 --dim-nl 8 --record dim
# python run.py --obstacle-shape 6 --dim-internal 4 --dim-nl 8 --record dim
# python run.py --obstacle-shape 6 --dim-internal 8 --dim-nl 2 --record dim
# python run.py --obstacle-shape 6 --dim-internal 8 --dim-nl 4 --record dim

# for dim_internal in 8 10 12 14; do
#     for dim_nl in 8 10 12 14; do
#         echo "Running simulation with dim-internal: $dim_internal and dim-nl: $dim_nl"
#         python run.py --obstacle-shape 6 --dim-internal "$dim_internal" --dim-nl "$dim_nl" --record dim --filename dim_rt
#         echo "Finished running with dim-internal: $dim_internal and dim-nl: $dim_nl"
#         echo "------------------------------------"
#     done
# done




echo "All sensitivity analysis simulations completed."