#! /bin/sh
# Code adapted from: Equivariant Hypergraph Neural Networks, 2022,
# Jinwoo Kim and Saeyoon Oh and Sungjun Cho and Seunghoon Hong.
# Available from: https://github.com/jw9730/ehnn
# Article: https://arxiv.org/abs/2208.10428
# Accessed: 30 September 2023
ehnn_n_layers=2
ehnn_hidden_channel=256
ehnn_n_heads=1
ehnn_hyper_layers=3
ehnn_hyper_dropout=0
ehnn_input_dropout=0
ehnn_mlp_classifier="True"
Classifier_num_layers=1
Classifier_hidden=128
normalization='ln'
lr=0.001
dropout=0
ehnn_att0_dropout=0
ehnn_att1_dropout=0
wd=0
folder=$1
epochs=$2
warmup_epochs=$3
ehnn_only=$4
runs=$5



for lr in 0.001 0.0001
do
    for wd in 0
    do
        for ehnn_n_heads in 8 4
        do
            for ehnn_hyper_dropout in 0
            do
                for ehnn_hidden_channel in 128 256
                do
                  ehnn_inner_channel=$ehnn_hidden_channel
                  ehnn_qk_channel=$ehnn_hidden_channel
                  ehnn_pe_dim=$ehnn_hidden_channel
                  ehnn_hyper_dim=$ehnn_hidden_channel

                  bash run_one_model.sh $ehnn_n_layers $ehnn_hidden_channel $ehnn_inner_channel \
                  $ehnn_qk_channel $ehnn_n_heads $ehnn_pe_dim $ehnn_hyper_dim $ehnn_hyper_layers $ehnn_hyper_dropout \
                  $ehnn_input_dropout $lr $dropout $ehnn_mlp_classifier $wd \
                  $Classifier_hidden $Classifier_num_layers $normalization $ehnn_att0_dropout $ehnn_att1_dropout \
                  $folder $epochs $warmup_epochs $ehnn_only $runs
                done
            done
        done
    done
done

