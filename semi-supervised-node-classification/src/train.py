"""
Code adapted from two sources:

 Equivariant Hypergraph Neural Networks, 2022,
 Jinwoo Kim and Saeyoon Oh and Sungjun Cho and Seunghoon Hong.
 Available from: https://github.com/jw9730/ehnn
 Article: https://arxiv.org/abs/2208.10428
 Accessed: 30 September 2023

 and

 Code adapted from: Counterfactual and Factual Reasoning over Hypergraphs for Interpretable Clinical Predictions on EHR, 2022,
 Xu, Ran and Yu, Yue and Zhang, Chao and Ali, Mohammed K and Ho, Joyce C and Yang, Carl
 Available from: https://github.com/ritaranx/CACHE
 Article: https://proceedings.mlr.press/v193/xu22a.html
 Accessed: 4 October 2023
"""

import argparse
import os
import os.path as osp
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

from convert_datasets_to_pygDataset import dataset_Hypergraph
from ehnn.mask import build_mask, build_mask_chunk
from ehnn_classifier import EHNNClassifier
from preprocessing import ConstructH, ConstructHSparse, ExtractV2E, rand_train_test_idx
from subsetlearner.models import ViewLearner
import random


def count_parameters(cp_model):
    return sum(p.numel() for p in cp_model.parameters() if p.requires_grad)


def gumbel_softmax(w_e_v, gs_device, temp):
    # gumbel softmax
    bias = 0.0 + 0.0001  # If bias is 0, we run into problems
    eps = (bias - (1 - bias)) * torch.rand(w_e_v.size()) + (1 - bias)
    gate_inputs = torch.log(eps) - torch.log(1 - eps)
    gate_inputs = gate_inputs.to(gs_device)
    gate_inputs = (gate_inputs + w_e_v) / temp
    gs_p_e_v = torch.sigmoid(gate_inputs).squeeze()
    return gs_p_e_v


def get_f_and_cf_predictions(pe_v, ehnn_model, fcf_data, fcf_ehnn_cache, fcf_device, fcf_args):
    # G' preserve nodes in edges where ^pe,v > gamma
    zeros = torch.tensor(0.0).to(fcf_device)
    ones = torch.tensor(1.0).to(fcf_device)

    # classified nodes in the evaluate task
    # value form AUPR and a lower to compare
    factual_g = torch.where(pe_v <= fcf_args.threshold, zeros, ones)
    factual_g.to(fcf_device)
    # G/G' preserve nodes in edges where ^pe,v > 0.5
    counterfactual_g = torch.where(pe_v <= fcf_args.threshold, ones, zeros)
    percentage_preserved_nodes = ((factual_g == 1).sum(dim=0)).int().item() / factual_g.shape[0]
    print(f'% preserved nodes: {percentage_preserved_nodes}')
    # factual prediction
    factual_node_logits, _ = ehnn_model(fcf_data, fcf_ehnn_cache, augmented_g=factual_g)
    fcf_factual_node_predictions = torch.sigmoid(factual_node_logits.squeeze())

    # counterfactual prediction
    counter_factual_node_logits, _ = ehnn_model(fcf_data, fcf_ehnn_cache, augmented_g=counterfactual_g)
    fcf_counter_factual_node_predictions = torch.sigmoid(counter_factual_node_logits.squeeze())

    return fcf_factual_node_predictions, fcf_counter_factual_node_predictions, factual_node_logits.squeeze(), counter_factual_node_logits.squeeze(), percentage_preserved_nodes


def get_f_and_cf_loss(np_sigmoid, fnp, cfnp, gamma, view_alpha):
    # factual loss
    coef = np_sigmoid.detach().clone()
    coef[np_sigmoid >= 0.5] = 1
    coef[np_sigmoid < 0.5] = -1
    loss_f = torch.mean(torch.clamp(torch.add(coef * (0 - fnp), gamma), min=0))

    # counter factual loss
    coef = np_sigmoid.detach().clone()
    coef[np_sigmoid >= 0.5] = -1
    coef[np_sigmoid < 0.5] = 1
    loss_cf = torch.mean(torch.clamp(torch.add(coef * (0 - cfnp), gamma), min=0))

    # factual and counterfactual view loss
    f_and_cf_loss = view_alpha * loss_f + (1 - view_alpha) * loss_cf
    return f_and_cf_loss


@torch.no_grad()
def evaluate(e_model, e_view_learner, e_ehnn_cache, e_data, e_split_idx,
             e_epoch, e_args, e_results, e_device, e_duration, e_run, e_model_loss, e_valid_loss, e_vl_loss):
    def eval_func(y_true, y_logits, apply_sigmoid=True):
        y_true = y_true.detach().cpu().numpy()
        if apply_sigmoid:
            y_pred = (torch.sigmoid(y_logits) >= e_args.threshold).int().detach().cpu().numpy()
        else:
            y_pred = (y_logits >= e_args.threshold).int().detach().cpu().numpy()
        y_scores = torch.sigmoid(y_logits).detach().cpu().numpy()

        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)

        return accuracy, roc_auc, aupr, f1_macro

    e_model.eval()
    # use original graph (G)
    with torch.no_grad():
        e_node_logits, _ = e_model(e_data, e_ehnn_cache)
    # sigmoid is applied in the eval function
    e_node_predictions = e_node_logits.squeeze()

    train_acc_g, train_auc_g, train_aupr_g, train_f1_macro_g = \
        eval_func(e_data.y[e_split_idx['train']], e_node_predictions[e_split_idx['train']])

    valid_acc_g, valid_auc_g, valid_aupr_g, valid_f1_macro_g = \
        eval_func(e_data.y[e_split_idx['valid']], e_node_predictions[e_split_idx['valid']])

    test_acc_g, test_auc_g, test_aupr_g, test_f1_macro_g = \
        eval_func(e_data.y[e_split_idx['test']], e_node_predictions[e_split_idx['test']])

    grid_path = f'{e_args.model_lr}-{e_args.model_wd}-{e_args.ehnn_n_heads}-{e_args.ehnn_hyper_dropout}-{e_args.ehnn_hidden_channel}'
    os.makedirs(f'exp_result/saved_models/', exist_ok=True)
    os.makedirs(f'exp_result/{e_args.folder}/{grid_path}/run_{e_run}/', exist_ok=True)

    if valid_f1_macro_g >= e_results['valid_f1']:
        print(f'saving results with valid_f1 {valid_f1_macro_g}')
        e_results['valid_f1'] = valid_f1_macro_g
        e_results['test_f1'] = test_f1_macro_g
        e_results['valid_aupr'] = valid_aupr_g
        e_results['test_aupr'] = test_aupr_g
        e_results['valid_acc'] = valid_acc_g
        e_results['test_acc'] = test_acc_g
        e_results['valid_roc'] = valid_auc_g
        e_results['test_roc'] = test_auc_g
        e_results['epoch'] = e_epoch

        y_true = e_data.y[e_split_idx['valid']].detach().cpu().numpy()
        y_pred = (torch.sigmoid(e_node_predictions[e_split_idx['valid']])).detach().cpu().numpy()

        pd.DataFrame(y_true).to_pickle(f'exp_result/{e_args.folder}/{grid_path}/y_true.pkl')
        pd.DataFrame(y_pred).to_pickle(f'exp_result/{e_args.folder}/{grid_path}/y_pred.pkl')

    if e_epoch < e_args.warmup_epochs:
        train_acc_gf, train_auc_gf, train_aupr_gf, train_f1_macro_gf, train_recall_gf, \
        valid_acc_gf, valid_auc_gf, valid_aupr_gf, valid_f1_macro_gf, valid_recall_gf, \
        test_acc_gf, test_auc_gf, test_aupr_gf, test_f1_macro_gf, test_recall_gf, \
        train_acc_gcf, train_auc_gcf, train_aupr_gcf, train_f1_macro_gcf, train_recall_gcf,\
        valid_acc_gcf, valid_auc_gcf, valid_aupr_gcf, valid_f1_macro_gcf, valid_recall_gcf,\
        test_acc_gcf, test_auc_gcf, test_aupr_gcf, test_f1_macro_gcf, test_recall_gcf, \
        e_pc_preserved_nodes, vl_loss = \
            0, 0, 0, 0, 0, \
            0, 0, 0, 0, 0, \
            0, 0, 0, 0, 0, \
            0, 0, 0, 0, 0, \
            0, 0, 0, 0, 0, \
            0, 0, 0, 0, 0, \
            0, 0
    else:
        # get the edge weight
        e_view_learner.eval()

        with torch.no_grad():
            e_weights_ev = e_view_learner(e_model, data, ehnn_cache)

        e_p_e_v = gumbel_softmax(e_weights_ev, e_device, e_args.temperature)

        _, _, e_factual_node_predictions, e_counter_factual_node_predictions, e_pc_preserved_nodes = \
            get_f_and_cf_predictions(e_p_e_v, e_model, e_data, e_ehnn_cache, e_device, e_args)

        e_num_nodes = e_data.n_x[0] if isinstance(e_data.n_x, list) else e_data.n_x

        if valid_f1_macro_g >= e_results['valid_f1']:
            get_subset_ranking(e_data.edge_index, e_num_nodes, e_data, e_args, e_run, grid_path,
                               e_node_predictions, e_weights_ev)

        train_acc_gf, train_auc_gf, train_aupr_gf, train_f1_macro_gf = eval_func(
            e_data.y[e_split_idx['train']],
            e_factual_node_predictions[e_split_idx['train']], False)
        valid_acc_gf, valid_auc_gf, valid_aupr_gf, valid_f1_macro_gf = eval_func(
            e_data.y[e_split_idx['valid']],
            e_factual_node_predictions[e_split_idx['valid']], False)
        test_acc_gf, test_auc_gf, test_aupr_gf, test_f1_macro_gf = eval_func(
            e_data.y[e_split_idx['test']], e_factual_node_predictions[e_split_idx['test']], False)

        train_acc_gcf, train_auc_gcf, train_aupr_gcf, train_f1_macro_gcf = eval_func(
            e_data.y[e_split_idx['train']],
            e_counter_factual_node_predictions[e_split_idx['train']], False)
        valid_acc_gcf, valid_auc_gcf, valid_aupr_gcf, valid_f1_macro_gcf = eval_func(
            e_data.y[e_split_idx['valid']],
            e_counter_factual_node_predictions[e_split_idx['valid']], False)
        test_acc_gcf, test_auc_gcf, test_aupr_gcf, test_f1_macro_gcf = eval_func(
            e_data.y[e_split_idx['test']], e_counter_factual_node_predictions[e_split_idx['test']], )

    fname_train = f'exp_result/{e_args.folder}/{grid_path}/run_{e_run}/train.csv'
    fname_test = f'exp_result/{e_args.folder}/{grid_path}/run_{e_run}/test.csv'
    fname_valid = f'exp_result/{e_args.folder}/{grid_path}/run_{e_run}/valid.csv'

    if e_epoch == 0:
        # write csv headers on first epoch
        with open(fname_train, 'a+', encoding='utf-8') as f:
            f.write(
                'Epoch, duration,'
                'ACC_G, AUC_G, AUPR_G, F1_MACRO_G,'
                'ACC_Gf, AUC_Gf, AUPR_Gf, F1_MACRO_Gf,'
                'ACC_Gcf, AUC_Gcf, AUPR_Gcf, F1_MACRO_Gcf,'
                'Model Loss, VL Loss, % Pres Nodes\n')
        with open(fname_test, 'a+', encoding='utf-8') as f:
            f.write(
                'Epoch, duration,'
                'ACC_G, AUC_G, AUPR_G, F1_MACRO_G,'
                'ACC_Gf, AUC_Gf, AUPR_Gf, F1_MACRO_Gf,'
                'ACC_Gcf, AUC_Gcf, AUPR_Gcf, F1_MACRO_Gcf, % Pres Nodes\n')
        with open(fname_valid, 'a+', encoding='utf-8') as f:
            f.write(
                'Epoch, duration,'
                'ACC_G, AUC_G, AUPR_G, F1_MACRO_G,'
                'ACC_Gf, AUC_Gf, AUPR_Gf, F1_MACRO_Gf,'
                'ACC_Gcf, AUC_Gcf, AUPR_Gcf, F1_MACRO_Gcf,'
                'Valid Loss,  % Pres Nodes\n')

    # write values
    with open(fname_train, 'a+', encoding='utf-8') as f:
        f.write(f'{epoch + 1}, {e_duration:.3f},'
                f'{train_acc_g:.5f},{train_auc_g:.5f},{train_aupr_g:.5f},{train_f1_macro_g:.5f},'
                f'{train_acc_gf:.5f},{train_auc_gf:.5f}, {train_aupr_gf:.5f}, {train_f1_macro_gf:.5f},'
                f'{train_acc_gcf:.5f},{train_auc_gcf:.5f}, {train_aupr_gcf:.5f}, {train_f1_macro_gcf:.5f},'
                f'{e_model_loss:.5f}, {e_vl_loss:.5f}, {e_pc_preserved_nodes}\n')

    with open(fname_test, 'a+', encoding='utf-8') as f:
        f.write(f'{epoch + 1}, {e_duration:.3f},'
                f'{test_acc_g:.5f},{test_auc_g:.5f},{test_aupr_g:.5f},{test_f1_macro_g:.5f},'
                f'{test_acc_gf:.5f},{test_auc_gf:.5f}, {test_aupr_gf:.5f}, {test_f1_macro_gf:.5f},'
                f'{test_acc_gcf:.5f},{test_auc_gcf:.5f}, {test_aupr_gcf:.5f}, {test_f1_macro_gcf:.5f},'
                f'{e_pc_preserved_nodes}\n')

    with open(fname_valid, 'a+', encoding='utf-8') as f:
        f.write(f'{e_epoch + 1}, {e_duration:.3f},'
                f'{valid_acc_g:.5f},{valid_auc_g:.5f},{valid_aupr_g:.5f},{valid_f1_macro_g:.5f},'
                f'{valid_acc_gf:.5f},{valid_auc_gf:.5f}, {valid_aupr_gf:.5f}, {valid_f1_macro_gf:.5f},'
                f'{valid_acc_gcf:.5f},{valid_auc_gcf:.5f}, {valid_aupr_gcf:.5f}, {valid_f1_macro_gcf:.5f},'
                f'{e_valid_loss:.5f}, {e_pc_preserved_nodes}\n')

    # print validation set as we go
    print(f'Epoch {e_epoch + 1}, {e_duration:.2f}s,'
          f'vag: {valid_acc_g:.2f}, f1g: {valid_f1_macro_g:.2f}, '
          f'vagf: {valid_acc_gf:.2f}, f1gf: {valid_f1_macro_gf:.2f}, '
          f'vagcf: {valid_acc_gcf:.2f}, f1gcf: {valid_f1_macro_gcf:.2f}, '
          f'MLoss: {e_model_loss:.5f}, VLLoss: {e_vl_loss:.5f}, %pn: {e_pc_preserved_nodes}')

    if e_epoch == e_args.epochs - 1:
        # capture best across multiple runs
        if not os.path.exists('exp_result/multi-hg-res'):
            os.makedirs('exp_result/multi-hg-res')
            with open(f'exp_result/multi-hg-res/multi-runs.csv', 'a+', encoding='utf-8') as f:
                f.write('Data,LR,WD,SAHs,DO,HCLs,run,'
                        'Best Valid F1,Test F1,Valid AUPR,Test AUPR,'
                        'Valid Acc,Test Acc,Valid ROC,Test ROC,Best Valid F1 epoch,\n')

        with open(f'exp_result/multi-hg-res/multi-runs.csv', 'a+', encoding='utf-8') as f:
            f.write(f'{e_args.folder},{e_args.model_lr},{e_args.model_wd},{e_args.ehnn_n_heads},{e_args.ehnn_hyper_dropout},{e_args.ehnn_hidden_channel},{run},'
                    f'{e_results["valid_f1"]},{e_results["test_f1"]}, {e_results["valid_aupr"]},{e_results["test_aupr"]},'
                    f'{e_results["valid_acc"]},{e_results["test_acc"]},{e_results["valid_roc"]},{e_results["test_roc"] },{e_results["epoch"]},\n')


def get_subset_ranking(sr_edge_index, sr_num_nodes, sr_data, sr_args, sr_run, sr_grid_path, sr_node_predictions, sr_weights_ev):
    edge_index_clone = sr_edge_index.copy()
    num_e = (data.num_hyperedges[0] if isinstance(data.num_hyperedges, list) else data.num_hyperedges)

    y_pred = (torch.sigmoid(sr_node_predictions) >= sr_args.threshold).int().detach().cpu().numpy()
    y_true = sr_data.y.detach().cpu().numpy()

    best_weights_sigmoid = torch.sigmoid(sr_weights_ev)
    best_weights_ev_clone = sr_weights_ev.reshape(1, -1).clone().detach().to('cpu').numpy()
    best_weights_sigmoid_clone = best_weights_sigmoid.reshape(1, -1).clone().detach().to('cpu').numpy()
    # concat into a 4 x N array [node, edge, weight, weight sigmoid]
    index_weight_concat = np.concatenate((edge_index_clone, best_weights_ev_clone, best_weights_sigmoid_clone), axis=0)
    # sort whole array by weight descending
    index_weight_concat = index_weight_concat[:, index_weight_concat[2, :].argsort()[::-1]]
    # populate a dict with node id -> edge membership
    node_dict = {}
    node_result_dict = {}
    edge_pos_dict = {}
    edge_neg_dict = {}
    # record the prediction for each node
    for i in range(sr_num_nodes):
        node_dict[i] = []
        node_result_dict[i] = {}
        node_result_dict[i]['y_pred'] = y_pred[i]
        node_result_dict[i]['y_true'] = y_true[i]
        node_result_dict[i]['y_correct'] = int(y_pred[i] == y_true[i])
        node_result_dict[i]['edge_weights'] = {}
    # record the edge weight sigmoid for each node-edge relationship
    for i in range(index_weight_concat.shape[1]):
        edge = int(index_weight_concat[1][i])
        node_result_dict[index_weight_concat[0][i]]['edge_weights'][edge] = index_weight_concat[3][i]
        node_dict[index_weight_concat[0][i]].append(index_weight_concat[1][i])

    # Popoulate edge weights dictionaries
    for e in range(num_e):
        edge_pos_dict[e] = []
        edge_neg_dict[e] = []
    for i in range(index_weight_concat.shape[1]):
        node = index_weight_concat[0][i]
        if node_result_dict[node]['y_true'] == 1:
            edge_pos_dict[index_weight_concat[1][i]].append(index_weight_concat[3][i])
        else:
            edge_neg_dict[index_weight_concat[1][i]].append(index_weight_concat[3][i])

    sorted_node_dict = dict(sorted(node_dict.items()))

    # record top x % of edge weights for each node
    with open(f"exp_result/{sr_args.folder}/{sr_grid_path}/run_{sr_run}/nodes_deleted.csv", "w") as f_del, \
            open(f"exp_result/{sr_args.folder}/{sr_grid_path}/run_{sr_run}/nodes_remain.csv", "w") as f_rem:
        f_rem.write('Node,Label,Prediction,Correct,\n')
        f_del.write('Node,Label,Prediction,Correct\n,')
        for key, values in sorted_node_dict.items():
            f_rem.write(f'{key},{node_result_dict[key]["y_true"]},{node_result_dict[key]["y_true"]},{node_result_dict[key]["y_correct"]},')
            f_del.write(f'{key},{node_result_dict[key]["y_true"]},{node_result_dict[key]["y_true"]},{node_result_dict[key]["y_correct"]},')
            rem_size = int(len(values) * sr_args.remain_percentage)
            if rem_size < 5 and len(values) >= 5:
                rem_size = 5
            elif rem_size < 5 and len(values) < 5:
                rem_size = len(values)
            remain = [str(int(x)) for x in values[:rem_size]]
            f_rem.write(",".join(remain))
            f_rem.write('\n')
            delete = [str(int(x)) for x in values[rem_size:]]
            f_del.write(",".join(delete))
            f_del.write('\n')

    # record aggregate node-edge weights for each edge
    with open(f"exp_result/{sr_args.folder}/{sr_grid_path}/run_{sr_run}/positive_edge_results.csv", "w") as f_pos:
        for key, value in edge_pos_dict.items():
            f_pos.write(str(key) + ',')
            for x in value:
                f_pos.write(f"{x},")
            f_pos.write("\n")

    with open(f"exp_result/{sr_args.folder}/{sr_grid_path}/run_{sr_run}/negative_edge_results.csv", "w") as f_neg:
        for key, value in edge_neg_dict.items():
            f_neg.write(str(key) + ',')
            for x in value:
                f_neg.write(f"{x},")
            f_neg.write("\n")

    # record all node-edge weights in a single place
    with open(f"exp_result/{sr_args.folder}/{sr_grid_path}/run_{sr_run}/all_node_edge_weights.csv", "w") as f_all:
        # Write headers
        f_all.write('Node,Label,Prediction,Correct,')
        for e in range(num_e):
            f_all.write(f'{e},')
        f_all.write('\n')
        # write vals
        for node in range(num_nodes):
            f_all.write(f'{node},{node_result_dict[node]["y_true"]},{node_result_dict[node]["y_pred"]},{node_result_dict[node]["y_correct"]},')
            for edge in range(num_e):
                if edge in node_result_dict[node]["edge_weights"]:
                    f_all.write(f'{node_result_dict[node]["edge_weights"][edge]},')
                else:
                    f_all.write('x,')
            f_all.write('\n')


if __name__ == "__main__":
    # --- Main part of the training ---
    # Parse arguments
    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument("--folder", type=str)
    parser.add_argument("--ehnn_only", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--train_prop", type=float, default=0.6)
    parser.add_argument("--valid_prop", type=float, default=0.5)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--dropout", default=0.75, type=float)
    parser.add_argument("--threshold", default=0.5, type=float)

    # Interpretable Subset Module Parameters
    parser.add_argument('--model_lr', default=1e-3, type=float)
    parser.add_argument('--model_wd', default=1e-3, type=float)
    parser.add_argument('--view_lr', default=1e-3, type=float)
    parser.add_argument('--view_wd', default=0, type=float)
    parser.add_argument("--Classifier_num_layers", default=2, type=int)  # How many layers of decoder
    parser.add_argument("--Classifier_hidden", default=64, type=int)  # Decoder hidden units
    parser.add_argument('--view_alpha', type=float, default=0.5)
    parser.add_argument('--view_lambda', type=float, default=0.00000001)
    parser.add_argument('--model_lambda', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.5)  # temperature for gumbel softmax - lower is stricter
    parser.add_argument('--remain_percentage', default=0.3, type=float)

    # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument("--normalization", default="ln")
    # Args for EHNN
    parser.add_argument("--ehnn_n_layers", type=int, default=2, help="layer num")
    parser.add_argument("--ehnn_ff_layers", type=int, default=2, help="encoder ff layer num")
    parser.add_argument("--ehnn_qk_channel", type=int, default=64, help="qk channel")
    parser.add_argument("--ehnn_n_heads", type=int, default=4, help="n_heads")
    parser.add_argument("--ehnn_inner_channel", type=int, default=64, help="inner channel")
    parser.add_argument("--ehnn_hidden_channel", type=int, default=64, help="hidden dim")
    parser.add_argument("--ehnn_pe_dim", type=int, default=64, help="pe dim")
    parser.add_argument("--ehnn_hyper_dim", type=int, default=64, help="hypernetwork dim")
    parser.add_argument("--ehnn_hyper_layers", type=int, default=2, help="hypernetwork layers")
    parser.add_argument("--ehnn_hyper_dropout", type=float, default=0.2, help="hypernetwork dropout rate", )
    parser.add_argument("--ehnn_input_dropout", type=float, default=0.0, help="input dropout rate")
    parser.add_argument("--ehnn_mlp_classifier", type=str, help="mlp classifier head")
    parser.add_argument("--ehnn_att0_dropout", type=float, default=0.0, help="att0 dropout rate")
    parser.add_argument("--ehnn_att1_dropout", type=float, default=0.0, help="att1 dropout rate")
    parser.add_argument("--warmup_epochs", type=int, default=10)

    args = parser.parse_args()

    # remove random noise by seeding all randomnness
    SEED = 14
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)  # if using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # -- START SECTION -- #
    # Load and preprocess data

    # convert data to pytorch geometric data object
    dataset = dataset_Hypergraph(folder=args.folder)
    data = dataset.data
    args.num_features = dataset.num_features
    args.num_classes = 1
    print(f'num nodes: {data.n_x}')
    # -- END SECTION -- #

    # -- START SECTION -- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ehnn_cache = None

    # Extract only edges from vertices to edges (V2E) in the edge_index
    data = ExtractV2E(data)  # [2, |E|] (caution: [V; E])
    x = data.x  # [N, D] (nodes and feature)
    y = data.y  # [N,] (labels)
    num_nodes = data.n_x[0] if isinstance(data.n_x, list) else data.n_x
    num_hyperedges = (data.num_hyperedges[0] if isinstance(data.num_hyperedges, list) else data.num_hyperedges)

    assert (num_nodes + num_hyperedges - 1) == data.edge_index[1].max().item(), "num_hyperedges does not match!"
    assert num_nodes == data.x.size(0), f"num_nodes does not match!"

    original_edge_index = data.edge_index
    data.original_edge_index = original_edge_index
    data = ConstructH(data)  # [|V|, |E|]
    # dense incidence matrix
    incidence_d = (
        torch.tensor(data.edge_index, dtype=torch.float32)
        .to_sparse(2)
        .coalesce()
        .to(device)
    )
    edge_orders_d = (
        torch.sparse.sum(incidence_d, 0).to_dense().long().to(device)
    )  # [|E|,]

    # sparse incidence matrix
    data.edge_index = original_edge_index
    data = ConstructHSparse(data)
    num_nodes = data.x.shape[0]
    num_hyperedges = np.max(data.edge_index[1]) + 1
    incidence_s = torch.sparse_coo_tensor(
        data.edge_index,
        torch.ones(len(data.edge_index[0])),
        (num_nodes, num_hyperedges),
        device=device,
    ).coalesce()
    edge_orders_s = (
        torch.sparse.sum(incidence_s, 0).to_dense().long().to(device)
    )  # [|E|,]

    assert (incidence_d.indices() - incidence_s.indices() == 0).all()
    assert (incidence_d.values() - incidence_s.values() == 0).all()

    incidence = incidence_d
    edge_orders = edge_orders_d

    os.makedirs(f"./cache/{args.folder}", exist_ok=True)
    if not osp.isfile(f"./cache/{args.folder}/nacc.pt"):
        print(f"preprocessing nacc")
        prefix_normalizer = (torch.sparse.sum(incidence, 0).to_dense().to(device))  # [|E|,]
        prefix_normalizer = prefix_normalizer.masked_fill_(prefix_normalizer == 0, 1e-5)
        suffix_normalizer = (torch.sparse.sum(incidence, 1).to_dense().to(device))  # [|V|,]
        suffix_normalizer = suffix_normalizer.masked_fill_(suffix_normalizer == 0, 1e-5)

        # chunked mask computation
        mask_dict_chunk = build_mask_chunk(incidence, device)  # Dict(overlap: [|E|, |E|] sparse)
        overlaps_chunk, masks_chunk = list(mask_dict_chunk.keys()), list(mask_dict_chunk.values())
        overlaps_chunk = torch.tensor(overlaps_chunk, dtype=torch.long, device=device)  # [|overlaps|,]
        masks_chunk = torch.stack(masks_chunk, dim=0).coalesce()  # [|overlaps|, |E|, |E|]]

        # correctness check with non-chunked masks
        mask_dict = build_mask(incidence, device)  # Dict(overlap: [|E|, |E|] sparse)
        overlaps, masks = list(mask_dict.keys()), list(mask_dict.values())
        overlaps = torch.tensor(overlaps, dtype=torch.long, device=device)  # [|overlaps|,]
        masks = torch.stack(masks, dim=0).coalesce()  # [|overlaps|, |E|, |E|]]
        assert (masks.indices() - masks_chunk.indices() == 0).all()
        assert (masks.values() - masks_chunk.values() == 0).all()
        assert (overlaps - overlaps_chunk == 0).all()

        masks = masks_chunk
        overlaps = overlaps_chunk
        n_overlaps = len(overlaps)

        normalizer = (torch.sparse.sum(masks, 2).to_dense().unsqueeze(-1))  # [|overlaps|, |E|, 1]
        normalizer = normalizer.masked_fill_(normalizer == 0, 1e-5)

        ehnn_cache = dict(
            incidence=incidence,
            edge_orders=edge_orders,
            overlaps=overlaps,
            n_overlaps=n_overlaps,
            prefix_normalizer=prefix_normalizer,
            suffix_normalizer=suffix_normalizer,
        )

        torch.save(ehnn_cache, f"./cache/{args.folder}/nacc.pt")
        print(f"saved ehnn_cache for nacc")
    else:
        ehnn_cache = torch.load(f"./cache/{args.folder}/nacc.pt")
    print(f"number of mask channels: {ehnn_cache['n_overlaps']}")

    # -- END SECTION -- #

    # Get splits
    split_idx = rand_train_test_idx(data.y)
    # Load model (f)
    model = EHNNClassifier(args, ehnn_cache)

    # Load int. subset learner (g)
    view_learner = ViewLearner( args.ehnn_hidden_channel, args.Classifier_hidden)

    # Put models on device
    model, view_learner, data = model.to(device), view_learner.to(device), data.to(device)
    print(f"num params: {count_parameters(model)}")

    # Main training and evaluation
    # Cost sensitive learning
    pos_count = (data.y == 1).sum(dim=0)
    learning_cost = 1 / (pos_count / data.y.shape[0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=learning_cost)
    view_loss = 0
    model_loss = 0

    for run in range(args.runs):
        # Training loop
        start_time = time.time()
        train_idx = split_idx["train"].to(device)
        valid_idx = split_idx["valid"].to(device)
        model.reset_parameters()
        model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr, weight_decay=args.model_wd)
        view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr, weight_decay=args.view_wd)

        # Track best performance
        best_result = {'valid_f1': 0}

        for epoch in range(args.epochs):
            start_time = time.time()
            if epoch > args.warmup_epochs:
                args.view_lambda *= 0.8

            # Warm up stage
            if epoch < args.warmup_epochs:
                model.train()
                model_optimizer.zero_grad()
                out, _ = model(data, ehnn_cache)  # [N, D]
                out = out.squeeze()

                model_loss = criterion(out[train_idx], data.y[train_idx].float())
                valid_loss = criterion(out[valid_idx], data.y[valid_idx].float())
                model_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                model_optimizer.step()
            else:
                # Alternate training
                """STEP ONE - TRAIN THE LEARNER"""
                view_learner.train()
                view_learner.zero_grad()
                model.eval()

                # get the model node predictions for use in loss calculation
                with torch.no_grad():
                    node_logits, _ = model(data, ehnn_cache)
                node_predictions = torch.sigmoid(node_logits.squeeze())

                weights_ev = view_learner(model, data, ehnn_cache)

                p_e_v = gumbel_softmax(weights_ev, device, args.temperature)

                factual_node_predictions, counter_factual_node_predictions, _, _, _ = \
                    get_f_and_cf_predictions(p_e_v, model, data, ehnn_cache, device, args)

                f_cf_loss = get_f_and_cf_loss(node_predictions,
                                              factual_node_predictions, counter_factual_node_predictions,
                                              args.gamma, args.view_alpha)

                view_loss = f_cf_loss + args.view_lambda * torch.mean(weights_ev)
                print(f"epoch: {epoch} max: {torch.max(weights_ev):.3f} mean: {torch.mean(weights_ev):.3f}"
                      f" view_loss: {view_loss:.3f} pevmax: {torch.max(p_e_v)} pev_mean: {torch.mean(p_e_v)}")

                view_loss.backward()
                torch.nn.utils.clip_grad_norm_(view_learner.parameters(), 1.0)
                view_optimizer.step()

                """STEP TWO - TRAIN THE MAIN MODEL"""
                model.train()
                model.zero_grad()
                view_learner.eval()

                node_logits, _ = model(data, ehnn_cache)
                # using BCEWithLogitLoss we don't need to apply sigmoid for loss calculation
                node_predictions = node_logits.squeeze()
                # but for our custom View Learner loss, we need to apply sigmoid
                node_predictions_sigmoid = torch.sigmoid(node_predictions)

                # learn the edge weight (augmentation policy)
                with torch.no_grad():
                    weights_ev = view_learner(model, data, ehnn_cache)

                p_e_v = gumbel_softmax(weights_ev, device, args.temperature)

                factual_node_predictions, counter_factual_node_predictions, _, _, _ = \
                    get_f_and_cf_predictions(p_e_v, model, data, ehnn_cache, device, args)

                f_cf_loss = get_f_and_cf_loss(node_predictions_sigmoid,
                                              factual_node_predictions, counter_factual_node_predictions,
                                              args.gamma, args.view_alpha)

                model_loss = criterion(node_predictions[train_idx].float(), data.y[train_idx].float()) +\
                             args.model_lambda * f_cf_loss
                valid_loss = criterion(node_predictions[valid_idx].float(), data.y[valid_idx].float()) +\
                             args.model_lambda * f_cf_loss
                model_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                model_optimizer.step()

            end_time = time.time()
            duration = end_time - start_time
            evaluate(model, view_learner, ehnn_cache, data, split_idx, epoch, args, best_result, device, duration, run, model_loss, valid_loss, view_loss)


    print("All done! Exit python code")
    quit()
