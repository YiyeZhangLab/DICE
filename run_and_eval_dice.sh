# setsid bash run_and_eval_dice.sh > /dev/null 2>&1 &


dataset="dice_time_series_top10"


n_hidden_fea=20
# n_hidden_fea=150

# K_clusters=2
# K_clusters=5
K_clusters=4
# K_clusters=3

n_iter=30
n_input_fea=50
n_dummy_demov_fea=60
batch_size=256

if [ ! -d "log/${dataset}" ]; then
    mkdir -p log/${dataset}
fi

if [ ! -d "output/${dataset}" ]; then
    mkdir -p output/${dataset}
fi

echo "Strating training"

# train
nohup python DICE.py --cuda 0 --init_AE_epoch 1 --n_hidden_fea  $n_hidden_fea --output_path "./output/${dataset}/" \
--input_path "./dataset/${dataset}/" --filename_train "datatrain.pkl" --filename_test "datavalid.pkl" \
--n_input_fea ${n_input_fea} --n_dummy_demov_fea ${n_dummy_demov_fea} --lstm_layer 1 --lr 0.0001 \
--K_clusters ${K_clusters} --iter ${n_iter} --epoch_in_iter 1 --lambda_AE 1.0 --lambda_classifier 1.0 \
--lambda_outcome 10.0 --lambda_p_value 1.0 --batch_size ${batch_size} > log/${dataset}/k${K_clusters}hn$n_hidden_fea.log 2>&1 &

pid=$!

while kill -0 $pid 2> /dev/null; do
    sleep 1
done

echo "Training completed, starting Architecture search"

# Architecture search
# nohup python NAS_DICE.py --cuda 0 --training_output_path "./output/${dataset}/" --input_path "./dataset/${dataset}/" \
# --filename_train datatrain.pkl --filename_valid datavalid.pkl --filename_test datatest.pkl \
# --n_input_fea ${n_input_fea} --n_dummy_demov_fea ${n_dummy_demov_fea} > log/${dataset}/NASk${K_clusters}hn${n_hidden_fea}.log 2>&1 &


pid=$!

while kill -0 $pid 2> /dev/null; do
    sleep 1
done

echo "Training completed, starting metrics calculation"

# metrics
nohup python clustering_metrics.py --cuda 0 --training_output_path "./output/${dataset}/" --input_path "./dataset/${dataset}/" \
--filename_train "datatrain.pkl" --filename_valid "datavalid.pkl" --filename_test "datatest.pkl" \
--n_input_fea ${n_input_fea} --n_dummy_demov_fea ${n_dummy_demov_fea} --K_clusters ${K_clusters} \
--n_hidden_fea ${n_hidden_fea} --batch_size ${batch_size} > log/${dataset}/metrics_k${K_clusters}_hn${n_hidden_fea}.log 2>&1 &

pid=$!

while kill -0 $pid 2> /dev/null; do
    sleep 1
done

echo "Metrics calculation completed, starting outcome prediction"

# outcome prediction
nohup python outcome_prediction.py --cuda 0 --training_output_path "./output/${dataset}/" --input_path "./dataset/${dataset}/" \
--filename_train "datatrain.pkl" --filename_valid "datavalid.pkl" --filename_test "datatest.pkl" \
--n_input_fea ${n_input_fea} --n_dummy_demov_fea ${n_dummy_demov_fea} --K_clusters ${K_clusters} \
--n_hidden_fea ${n_hidden_fea} --batch_size ${batch_size} > log/${dataset}/outcome_k${K_clusters}_hn${n_hidden_fea}.log 2>&1 &

pid=$!

while kill -0 $pid 2> /dev/null; do
    sleep 1
done

echo "Outcome prediction completed, starting visualization"

# viaual
nohup python representation_visualization.py --cuda 0 --training_output_path "./output/${dataset}/" \
--input_path "./dataset/${dataset}/" --filename_train "datatrain.pkl" --filename_valid "datavalid.pkl" --filename_test "datatest.pkl" \
--n_input_fea ${n_input_fea} --n_dummy_demov_fea ${n_dummy_demov_fea} --K_clusters ${K_clusters} \
--n_hidden_fea ${n_hidden_fea} --batch_size ${batch_size} > log/${dataset}/visual_k${K_clusters}_hn${n_hidden_fea}.log 2>&1 &

