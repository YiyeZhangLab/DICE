# setsid bash run_and_eval_dice.sh > /dev/null 2>&1 &


# dataset="ECG200"
# dataset="CricketX"
# dataset="person_4399427"
# dataset="top100_person_ed_only"
# dataset="06302025"
# dataset="06302025_fea20"
# dataset="07012025_fea100"
# dataset="07012025_fea10"
# dataset="mis_070225"
# dataset="mis_070825"
# dataset="mis_cat_250709"
# dataset="07142025_fea50_demo18"
# dataset="07172025_fea442_demo108"
# dataset="07182025_fea50_demo20"
dataset="07182025_fea442_demo110"
# dataset="mis_cat_250715"
# dataset="mis_250715"
# dataset="mis_cat_250716"


n_hidden_fea=20
# n_hidden_fea=30
# n_hidden_fea=40
# n_hidden_fea=70
# n_hidden_fea=100
# n_hidden_fea=150

# K_clusters=2
# K_clusters=5
K_clusters=4
# K_clusters=3

# n_iter=60
n_iter=30

if [ "${dataset}" == "ECG200" ]; then
    n_input_fea=1
    n_dummy_demov_fea=2
elif [ "${dataset}" == "CricketX" ]; then
    n_input_fea=1
    n_dummy_demov_fea=1
elif [ "${dataset}" == "breast_cancer" ]; then
    n_input_fea=30
    n_dummy_demov_fea=1
elif [ "${dataset}" == "person_4399427" ]; then
    n_input_fea=21
    n_dummy_demov_fea=5
elif [[ "${dataset}" == "top100_person_ed_only" || "${dataset}" == "06302025" ]]; then
    n_input_fea=60
    n_dummy_demov_fea=18
elif [ "${dataset}" == "06302025_fea20" ]; then
    n_input_fea=99
    n_dummy_demov_fea=28
elif [ "${dataset}" == "07012025_fea100" ]; then
    n_input_fea=419
    n_dummy_demov_fea=108
elif [ "${dataset}" == "07012025_fea10" ]; then
    n_input_fea=52
    n_dummy_demov_fea=18
elif [ "${dataset}" == "mis_070225" ]; then
    n_input_fea=54
    n_dummy_demov_fea=4
elif [ "${dataset}" == "mis_070825" ]; then
    n_input_fea=63
    n_dummy_demov_fea=14
elif [ "${dataset}" == "mis_cat_250709" ]; then
    n_input_fea=223
    n_dummy_demov_fea=14
elif [ "${dataset}" == "07142025_fea50_demo18" ]; then
    n_input_fea=50
    n_dummy_demov_fea=18
elif [ "${dataset}" == "mis_cat_250715" ]; then
    n_input_fea=220
    n_dummy_demov_fea=14
elif [ "${dataset}" == "mis_250715" ]; then
    n_input_fea=60
    n_dummy_demov_fea=14
elif [ "${dataset}" == "mis_cat_250716" ]; then
    n_input_fea=165
    n_dummy_demov_fea=14
elif [ "${dataset}" == "07172025_fea442_demo108" ]; then
    n_input_fea=442
    n_dummy_demov_fea=108
elif [ "${dataset}" == "07182025_fea50_demo20" ]; then
    n_input_fea=50
    n_dummy_demov_fea=20
elif [ "${dataset}" == "07182025_fea442_demo110" ]; then
    n_input_fea=442
    n_dummy_demov_fea=110
else
    echo "Dataset not supported: ${dataset}"
    exit 1
fi

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
--lambda_outcome 10.0 --lambda_p_value 1.0 > log/${dataset}/k${K_clusters}hn$n_hidden_fea.log 2>&1 &

# pid=$!

# while kill -0 $pid 2> /dev/null; do
#     sleep 1
# done

# echo "Training completed, starting Architecture search"

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
--n_hidden_fea ${n_hidden_fea} > log/${dataset}/metrics_k${K_clusters}_hn${n_hidden_fea}.log 2>&1 &

pid=$!

while kill -0 $pid 2> /dev/null; do
    sleep 1
done

echo "Metrics calculation completed, starting outcome prediction"

# outcome prediction
nohup python outcome_prediction.py --cuda 0 --training_output_path "./output/${dataset}/" --input_path "./dataset/${dataset}/" \
--filename_train "datatrain.pkl" --filename_valid "datavalid.pkl" --filename_test "datatest.pkl" \
--n_input_fea ${n_input_fea} --n_dummy_demov_fea ${n_dummy_demov_fea} --K_clusters ${K_clusters} \
--n_hidden_fea ${n_hidden_fea} > log/${dataset}/outcome_k${K_clusters}_hn${n_hidden_fea}.log 2>&1 &

pid=$!

while kill -0 $pid 2> /dev/null; do
    sleep 1
done

echo "Outcome prediction completed, starting visualization"

# viaual
nohup python representation_visualization.py --cuda 0 --training_output_path "./output/${dataset}/" \
--input_path "./dataset/${dataset}/" --filename_train "datatrain.pkl" --filename_valid "datavalid.pkl" --filename_test "datatest.pkl" \
--n_input_fea ${n_input_fea} --n_dummy_demov_fea ${n_dummy_demov_fea} --K_clusters ${K_clusters} \
--n_hidden_fea ${n_hidden_fea} > log/${dataset}/visual_k${K_clusters}_hn${n_hidden_fea}.log 2>&1 &

