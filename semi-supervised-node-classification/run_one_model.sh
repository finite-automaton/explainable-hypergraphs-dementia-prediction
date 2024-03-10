#! /bin/sh
# Code adapted from: Equivariant Hypergraph Neural Networks, 2022,
# Jinwoo Kim and Saeyoon Oh and Sungjun Cho and Seunghoon Hong.
# Available from: https://github.com/jw9730/ehnn
# Article: https://arxiv.org/abs/2208.10428
# Accessed: 30 September 2023

cd src || exit

ehnn_n_layers=$1
ehnn_hidden_channel=$2
ehnn_inner_channel=$3
ehnn_qk_channel=$4
ehnn_n_heads=$5
ehnn_pe_dim=$6
ehnn_hyper_dim="${7}"
ehnn_hyper_layers="${8}"
ehnn_hyper_dropout="${9}"
ehnn_input_dropout="${10}"
model_lr="${11}"
dropout="${12}"
ehnn_mlp_classifier="${13}"
model_wd="${14}"
Classifier_hidden="${15}"
Classifier_num_layers="${16}"
normalization="${17}"
ehnn_att0_dropout="${18}"
ehnn_att1_dropout="${19}"
folder="${20}"
epochs="${21}"
warmup_epochs="${22}"
ehnn_only="${23}"
runs="${24}"

echo "$PWD"
    echo =============
    echo ">>>> Start Training"
    python train.py \
        --ehnn_n_layers "$ehnn_n_layers" \
        --ehnn_hidden_channel "$ehnn_hidden_channel" \
        --ehnn_inner_channel "$ehnn_inner_channel" \
        --ehnn_qk_channel "$ehnn_qk_channel" \
        --ehnn_n_heads "$ehnn_n_heads" \
        --ehnn_pe_dim "$ehnn_pe_dim" \
        --ehnn_hyper_dim "$ehnn_hyper_dim" \
        --ehnn_hyper_layers "$ehnn_hyper_layers" \
        --ehnn_hyper_dropout "$ehnn_hyper_dropout" \
        --ehnn_input_dropout "$ehnn_input_dropout" \
        --model_lr "$model_lr" \
        --dropout "$dropout" \
        --model_wd "$model_wd" \
        --ehnn_mlp_classifier "$ehnn_mlp_classifier" \
        --Classifier_hidden "$Classifier_hidden" \
        --Classifier_num_layers "$Classifier_num_layers" \
        --normalization "$normalization" \
        --ehnn_att0_dropout "$ehnn_att0_dropout" \
        --ehnn_att1_dropout "$ehnn_att1_dropout" \
        --folder "$folder" \
        --epochs "$epochs" \
        --warmup_epochs "$warmup_epochs" \
        --ehnn_only "$ehnn_only" \
        --runs "$runs"

cd .. || exit
