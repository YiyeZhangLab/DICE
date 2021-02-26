#!/bin/bash
set -e
maxnhidden=102

# K=2
for (( n_hidden_fea = 20; n_hidden_fea< $maxnhidden; n_hidden_fea+=5)) 
do
    echo "process n_hidden_fea $n_hidden_fea for k2 "
	python DICE.py --init_AE_epoch 1 --n_hidden_fea  $n_hidden_fea --input_path "./dataset/" --filename_train "datatrain.pkl" --filename_test "datavalid.pkl" --n_input_fea 360 --n_dummy_demov_fea 9 --lstm_layer 1 --lr 0.0001 --K_clusters 2 --iter 60 --epoch_in_iter 1 --lambda_AE 1.0 --lambda_classifier 1.0 --lambda_outcome 10.0 --lambda_p_value 1.0 >k2hn$n_hidden_fea.log
done 

# K=3
for (( n_hidden_fea = 20; n_hidden_fea< $maxnhidden; n_hidden_fea+=5)) 
do
    echo "process n_hidden_fea $n_hidden_fea for k3 "
    python DICE.py  --init_AE_epoch 1 --n_hidden_fea  $n_hidden_fea --input_path "./dataset/" --filename_train "datatrain.pkl" --filename_test "datavalid.pkl" --n_input_fea 360 --n_dummy_demov_fea 9 --lstm_layer 1 --lr 0.0001 --K_clusters 3 --iter 60 --epoch_in_iter 1 --lambda_AE 1.0 --lambda_classifier 1.0 --lambda_outcome 10.0 --lambda_p_value 1.0 > k3hn$n_hidden_fea.log
done 

# K=4
for (( n_hidden_fea = 30; n_hidden_fea< $maxnhidden; n_hidden_fea+=5)) 
do
    echo "process n_hidden_fea $n_hidden_fea for k4 "
    python DICE.py  --init_AE_epoch 1 --n_hidden_fea  $n_hidden_fea --input_path "./dataset/" --filename_train "datatrain.pkl" --filename_test "datavalid.pkl" --n_input_fea 360 --n_dummy_demov_fea 9 --lstm_layer 1 --lr 0.0001 --K_clusters 4 --iter 60 --epoch_in_iter 1 --lambda_AE 1.0 --lambda_classifier 1.0 --lambda_outcome 10.0 --lambda_p_value 1.0 > k4hn$n_hidden_fea.log
done 

# K=5
for (( n_hidden_fea = 30; n_hidden_fea< $maxnhidden; n_hidden_fea+=5)) 
do
    echo "process n_hidden_fea $n_hidden_fea for k5 "
    python DICE.py  --init_AE_epoch 1 --n_hidden_fea  $n_hidden_fea --input_path "./dataset/" --filename_train "datatrain.pkl" --filename_test "datavalid.pkl" --n_input_fea 360 --n_dummy_demov_fea 9 --lstm_layer 1 --lr 0.0001 --K_clusters 5 --iter 60 --epoch_in_iter 1 --lambda_AE 1.0 --lambda_classifier 1.0 --lambda_outcome 10.0 --lambda_p_value 1.0 > k5hn$n_hidden_fea.log
done 

