#!/usr/bin/env bash

dataset="dice_time_series_top10_60_50"
n_iter=30
n_dummy_demov_fea=60
n_input_fea=50

batch_size=256
use_cuda=1

n_hidden_fea_list=(20 50 100 150)
K_clusters_list=(2 3 4 5 10)

mkdir -p "log/${dataset}" "output/${dataset}"

# Relaunch under nohup so runs survive terminal closes.
if [ -z "${DICE_MULTI_DETACHED}" ]; then
    export DICE_MULTI_DETACHED=1
    nohup bash "$0" "$@" > "log/${dataset}/multi_runner.log" 2>&1 &
    echo "Launched all experiments in background (pid $!). Monitor log/${dataset}/multi_runner.log"
    exit 0
fi

run_and_wait() {
    local log_file="$1"
    shift
    nohup "$@" > "${log_file}" 2>&1 &
    wait $!
}

for n_hidden_fea in "${n_hidden_fea_list[@]}"; do
    for K_clusters in "${K_clusters_list[@]}"; do
        echo "Starting training (K=${K_clusters}, hidden=${n_hidden_fea})"
        run_and_wait "log/${dataset}/k${K_clusters}hn${n_hidden_fea}.log" \
            python DICE.py --cuda ${use_cuda} --init_AE_epoch 1 --n_hidden_fea ${n_hidden_fea} --output_path "./output/${dataset}/" \
            --input_path "./dataset/${dataset}/" --filename_train "datatrain.pkl" --filename_test "datavalid.pkl" \
            --n_input_fea ${n_input_fea} --n_dummy_demov_fea ${n_dummy_demov_fea} --lstm_layer 1 --lr 0.0001 \
            --K_clusters ${K_clusters} --iter ${n_iter} --epoch_in_iter 1 --lambda_AE 1.0 --lambda_classifier 1.0 \
            --lambda_outcome 10.0 --lambda_p_value 1.0 --batch_size ${batch_size}

        echo "Training completed, starting metrics calculation (K=${K_clusters}, hidden=${n_hidden_fea})"
        run_and_wait "log/${dataset}/metrics_k${K_clusters}_hn${n_hidden_fea}.log" \
            python clustering_metrics.py --cuda ${use_cuda} --training_output_path "./output/${dataset}/" --input_path "./dataset/${dataset}/" \
            --filename_train "datatrain.pkl" --filename_valid "datavalid.pkl" --filename_test "datatest.pkl" \
            --n_input_fea ${n_input_fea} --n_dummy_demov_fea ${n_dummy_demov_fea} --K_clusters ${K_clusters} \
            --n_hidden_fea ${n_hidden_fea} --batch_size ${batch_size}

        echo "Metrics calculation completed, starting outcome prediction (K=${K_clusters}, hidden=${n_hidden_fea})"
        run_and_wait "log/${dataset}/outcome_k${K_clusters}_hn${n_hidden_fea}.log" \
            python outcome_prediction.py --cuda ${use_cuda} --training_output_path "./output/${dataset}/" --input_path "./dataset/${dataset}/" \
            --filename_train "datatrain.pkl" --filename_valid "datavalid.pkl" --filename_test "datatest.pkl" \
            --n_input_fea ${n_input_fea} --n_dummy_demov_fea ${n_dummy_demov_fea} --K_clusters ${K_clusters} \
            --n_hidden_fea ${n_hidden_fea} --batch_size ${batch_size}

        echo "Outcome prediction completed, starting visualization (K=${K_clusters}, hidden=${n_hidden_fea})"
        run_and_wait "log/${dataset}/visual_k${K_clusters}_hn${n_hidden_fea}.log" \
            python representation_visualization.py --cuda ${use_cuda} --training_output_path "./output/${dataset}/" \
            --input_path "./dataset/${dataset}/" --filename_train "datatrain.pkl" --filename_valid "datavalid.pkl" --filename_test "datatest.pkl" \
            --n_input_fea ${n_input_fea} --n_dummy_demov_fea ${n_dummy_demov_fea} --K_clusters ${K_clusters} \
            --n_hidden_fea ${n_hidden_fea} --batch_size ${batch_size}

        echo "Completed experiment (K=${K_clusters}, hidden=${n_hidden_fea})"

        output_dir="./output/${dataset}/hn_${n_hidden_fea}_K_${K_clusters}"
        if [ -d "${output_dir}" ]; then
            find "${output_dir}" -type f -name 'data_*.pickle' -delete
        fi
    done
done

echo "All experiments completed."
