import pathlib

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve, roc_curve, auc

# CONSTANTS
CDRGLOB_correspondence_dict = {0.0: "no impairment", 0.5: "questionable impairment", 1.0: "mild impairment",
                               2.0: "moderate impairment", 3.0: "severe impairment"}
EDUC_correspondence_dict = {1: "Less than high school", 2: "Up to High School", 3: "Bachelors degree", 4: "Masters degree", 5: "Doctorate"}
GENERIC_correspondence_dict = {
    0: "Absent",
    1: "Present",
    -4: "Unknown/Not available",
}
NACCNE4S_correspondence_dict = {
    0: "No e4 allele",
    1: "1 copy of e4 allele",
    2: "2 copies of e4 allele",
}

medication_descriptions = {
    "NACCAHTN": "Antihypertensive or blood pressure medication",
    "NACCHTNC": "Antihypertensive combination therapy",
    "NACCACEI": "Angiotensin converting enzyme (ACE) inhibitor",
    "NACCAAAS": "Antiadrenergic agent",
    "NACCBETA": "Beta-adrenergic blocking agent (Beta-Blocker)",
    "NACCCCBS": "Calcium channel blocking agent",
    "NACCDIUR": "Diuretic",
    "NACCVASD": "Vasodilator",
    "NACCANGI": "Angiotensin II inhibitor",
    "NACCLIPL": "Lipid lowering medication",
    "NACCNSD": "Nonsteroidal anti-inflammatory medication",
    "NACCAC": "Anticoagulant or antiplatelet agent",
    "NACCADEP": "Antidepressant",
    "NACCAPSY": "Antipsychotic agent",
    "NACCAANX": "Anxiolytic, sedative, or hypnotic agent",
    "NACCPDMD": "Antiparkinson agent",
    "NACCEMD": "Oestrogen hormone therapy",
    "NACCEPMD": "Oestrogen + progestin hormone therapy",
    "NACCDBMD": "Diabetes medication"
}

COMORBIDITIES_MAPPING = {
    "PARK": "Parkinson's disease",
    "BRNINJ": "Traumatic brain injury (TBI)",
    "HYCEPH": "Normal-pressure hydrocephalus (NPH)",
    "DEP": "Depression",
    "INCONTF": "Incontinence - bowel",
    "CVBYPASS": "Cardiac bypass procedure",
    "INCONTU": "Incontinence - urinary",
    "CVHATT": "Heart attack/cardiac arrest",
    "CVANGIO": "Angioplasty/endarterectomy/stent",
    "DIABETES": "Diabetes",
    "CVCHF": "Congestive heart failure",
    "PSYCDIS": "Other psychiatric disorder",
    "ALCOHOL": "Alcohol abuse",
    "CBSTROKE": "Stroke",
    "CVAFIB": "Atrial fibrillation",
    "SEIZURES": "Seizures",
    "THYROID": "Thyroid disease",
    "HYPERCHO": "Hypercholesterolemia",
    "NACCTBI": "History of traumatic brain injury (TBI)",
    "CBTIA": "Transient ischemic attack (TIA)",
    "B12DEF": "Vitamin B12 deficiency",
    "HEARING": "Hearing loss",
    "VISION": "Vision loss"
}


COMORBODITIES = ["PARK","BRNINJ","HYCEPH","DEP", "INCONTF","CVBYPASS","INCONTU","CVHATT","CVANGIO", "DIABETES", "CVCHF", "PSYCDIS", "ALCOHOL" "CBSTROKE", "CVAFIB", "SEIZURES", "CVOTHR","THYROID","HYPERCHO", "NACCTBI","CBTIA", "B12DEF"]

cohort_prefixes = ["2_visits", "3_visits", "4_visits", "5_visits", "6_visits", "7_visits", "8_visits", "9_visits", "10_visits", "11_visits"]
cohort_suffixes = ["ALL", "CDR", "COMORBODITIES", "DEPENDENCE", "FAMILY_AND_GENETICS", "LESS_300", "MEDICATION"]

GENERIC_labels = ["Absent", "Present", "Unknown/Not Available"]
GENERIC_order = [0, 1, -4]
generic_colors = ['red', 'green', 'blue']
raw_data_filepath = pathlib.Path('data/raw_data/investigator_ftldlbd_nacc61.csv')

raw_codes_filepath = pathlib.Path('data/raw_codes.csv')
matplotlib.rcParams['font.family'] = 'Times New Roman'

TITLE_SIZE = 24
LABEL_SIZE = 24
TICKS_SIZE = 20
figx = 12
figy = 8
SIZE_ADJUST = 8

cohort_data = 'hypergraph_data/4_visits_LESS_300_hg_input/final_rows.csv'


def plot_loss_comparison(df, filename, best_epoch):
    fig, ax = plt.subplots(figsize=(figx, figy))

    ax.plot(df['Epoch'], df['Model Loss'], label='Training Loss', color='blue', linewidth=3)
    ax.plot(df['Epoch'], df['Valid Loss'], label='Validation Loss', color='red', linewidth=3)

    ax.set_xlabel("Epoch", fontsize=LABEL_SIZE + 4)
    ax.set_ylabel("Loss", fontsize=LABEL_SIZE + 4)
    ax.legend(fontsize=LABEL_SIZE, title_fontsize=TITLE_SIZE)
    plt.xticks(fontsize=TICKS_SIZE)
    plt.yticks(fontsize=TICKS_SIZE)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.axvline(best_epoch, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'hypergraph_data/analysis/images/4yeargrid/{filename}-train-valid-loss.png', dpi=600)
    plt.show()


def plot_cohort_f1s():
    df = pd.read_csv('hypergraph_data/analysis/multi-best.csv')

    rename_dict = {
        "ALL": "All selected features",
        "LOW MISSING": "All selected features, filtered to low missing counts",
        "CDR": "CDR score features only",
        "COMORBIDITIES": "Comorbidity features only",
        "DEPENDENCE": "Dependence related features only",
        "FAMILY AND GENETICS": "Family dementia and genetic features only",
        "MEDICATION": "Medication features only"
    }
    df['Dataset'] = df['Dataset'].map(rename_dict)

    plt.style.use('seaborn-whitegrid')

    palette = plt.get_cmap('viridis')

    fig, axs = plt.subplots(4, 2, figsize=(14.5, 21))
    axs = axs.flatten()

    datasets = df['Dataset'].unique()
    for i, dataset in enumerate(datasets):
        subset = df[df['Dataset'] == dataset]

        axs[i].plot(subset['Year'], subset['Valid F1'], marker='', color=palette(0.3), linewidth=2.5, alpha=0.9, label='Validation Set F1 Scores')

        axs[i].plot(subset['Year'], subset['Test F1'], marker='', color=palette(0.6), linewidth=2.5, alpha=0.9, linestyle='dashed', label='Test Set F1 Scores')

        axs[i].set_xlabel("Year", fontsize=TICKS_SIZE)
        axs[i].set_ylabel("F1 Score", fontsize=TICKS_SIZE)

        axs[i].tick_params(axis='both', which='major', labelsize=TICKS_SIZE)

        axs[i].set_ylim(0, 0.8)
        axs[i].set_yticks([y / 100.0 for y in range(0, 81, 5)])

        axs[i].set_xlim(0, 10)
        axs[i].set_xticks(range(0, 11))

        axs[i].legend()

        axs[i].set_title(dataset, fontsize=LABEL_SIZE)

    plt.tight_layout()
    plt.savefig(f'hypergraph_data/analysis/images/multi/search.png', dpi=600)
    plt.show()


def plot_roc():
    y_pred = pd.read_pickle('hypergraph_data/analysis/best_base_preds/y_pred.pkl')[0].tolist()
    y_true = pd.read_pickle('hypergraph_data/analysis/best_base_preds/y_true.pkl')[0].tolist()

    plt.figure(figsize=(12, 8))
    display = RocCurveDisplay.from_predictions(y_true, y_pred, name='EHNN Base Model')
    display.plot()
    plt.grid(True)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=TICKS_SIZE - SIZE_ADJUST)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=TICKS_SIZE - SIZE_ADJUST)
    plt.xlabel('False Positive Rate', fontsize=LABEL_SIZE - SIZE_ADJUST)
    plt.ylabel('True Positive Rate', fontsize=LABEL_SIZE - SIZE_ADJUST)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--', label='No Skill')

    plt.plot([0, 0, 1], [0, 1, 1], color='green', linestyle='--', label='Perfect Skill')

    plt.legend(fontsize=TICKS_SIZE-SIZE_ADJUST)

    plt.savefig('hypergraph_data/analysis/best_base_preds/roc_base.png', dpi=600)
    plt.show()


def plot_aupr():
    y_pred = pd.read_pickle('hypergraph_data/analysis/best_base_preds/y_pred.pkl')[0].tolist()
    y_true = pd.read_pickle('hypergraph_data/analysis/best_base_preds/y_true.pkl')[0].tolist()

    plt.figure(figsize=(12, 8))
    display = PrecisionRecallDisplay.from_predictions(y_true, y_pred, name='EHNN Base Model')
    display.plot()
    plt.grid(True)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=TICKS_SIZE - SIZE_ADJUST)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=TICKS_SIZE - SIZE_ADJUST)
    plt.xlabel('Recall', fontsize=LABEL_SIZE - SIZE_ADJUST)
    plt.ylabel('Precision', fontsize=LABEL_SIZE - SIZE_ADJUST)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.axhline(y=1, color='green', linestyle='--', label='Perfect Skill')

    no_skill = sum(y_true) / len(y_true)
    plt.axhline(y=no_skill, color='orange', linestyle='--', label='No Skill')

    plt.legend(fontsize=TICKS_SIZE - SIZE_ADJUST)

    plt.savefig('hypergraph_data/analysis/best_base_preds/aupr_base.png', dpi=600)
    plt.show()


def plot_roc_comparison():
    y_pred_base = pd.read_pickle('hypergraph_data/analysis/best_base_preds/y_pred.pkl')[0].tolist()
    y_true_base = pd.read_pickle('hypergraph_data/analysis/best_base_preds/y_true.pkl')[0].tolist()

    y_pred_composite = pd.read_pickle('hypergraph_data/analysis/best_comp_preds/y_pred.pkl')[0].tolist()
    y_true_composite = pd.read_pickle('hypergraph_data/analysis/best_comp_preds/y_true.pkl')[0].tolist()

    plt.figure(figsize=(12, 8))

    fpr_base, tpr_base, _ = roc_curve(y_true_base, y_pred_base)
    roc_auc_base = auc(fpr_base, tpr_base)

    fpr_composite, tpr_composite, _ = roc_curve(y_true_composite, y_pred_composite)
    roc_auc_composite = auc(fpr_composite, tpr_composite)

    plt.plot(fpr_base, tpr_base, color='blue', label=f'EHNN Base Model (AUC = {roc_auc_base:.2f})')

    plt.plot(fpr_composite, tpr_composite, color='red', label=f'Composite Model (AUC = {roc_auc_composite:.2f})')

    plt.grid(True)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=TICKS_SIZE)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=TICKS_SIZE)
    plt.xlabel('False Positive Rate', fontsize=LABEL_SIZE + 4)
    plt.ylabel('True Positive Rate',  fontsize=LABEL_SIZE + 4)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.plot([0, 0, 1], [0, 1, 1], color='green', linestyle='--', label='Perfect Skill')
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--', label='No Skill')

    plt.legend(loc='lower right', fontsize=LABEL_SIZE)

    plt.savefig('hypergraph_data/analysis/best_comp_preds/roc_comp.png', dpi=600)
    plt.show()


def plot_aupr_comparison():
    y_pred_base = pd.read_pickle('hypergraph_data/analysis/best_base_preds/y_pred.pkl')[0].tolist()
    y_true_base = pd.read_pickle('hypergraph_data/analysis/best_base_preds/y_true.pkl')[0].tolist()

    y_pred_composite = pd.read_pickle('hypergraph_data/analysis/best_comp_preds/y_pred.pkl')[0].tolist()
    y_true_composite = pd.read_pickle('hypergraph_data/analysis/best_comp_preds/y_true.pkl')[0].tolist()

    plt.figure(figsize=(12, 8))

    precision_base, recall_base, _ = precision_recall_curve(y_true_base, y_pred_base)
    aupr_base = auc(recall_base, precision_base)

    precision_composite, recall_composite, _ = precision_recall_curve(y_true_composite, y_pred_composite)
    aupr_composite = auc(recall_composite, precision_composite)

    plt.plot(recall_base, precision_base, color='blue', label=f'EHNN Base Model (AUPR = {aupr_base:.2f})')

    plt.plot(recall_composite, precision_composite, color='red', label=f'Composite Model (AUPR = {aupr_composite:.2f})')

    plt.grid(True)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=TICKS_SIZE)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=TICKS_SIZE)
    plt.xlabel('Recall', fontsize=LABEL_SIZE + 4)
    plt.ylabel('Precision', fontsize=LABEL_SIZE + 4)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.axhline(y=1, color='green', linestyle='--', label='Perfect Skill')

    no_skill = sum(y_true_base) / len(y_true_base)
    plt.axhline(y=no_skill, color='orange', linestyle='--', label='No Skill')

    plt.legend(fontsize=LABEL_SIZE)

    plt.savefig('hypergraph_data/analysis/best_comp_preds/aupr_comp.png', dpi=600)
    plt.show()

def plot_custom_score_histogram():
    df = pd.read_csv('hypergraph_data/analysis/positive_edge_results-final.csv')

    bins = [i * 0.2 for i in range(11)]

    plt.figure(figsize=(12, 8))

    counts, bin_edges, _ = plt.hist(df['Custom Score'], bins=bins, edgecolor='black')

    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    for count, center in zip(counts, bin_centers):
        plt.text(center, count + max(counts) * 0.05, str(int(count)), ha='center', va='bottom', fontsize=LABEL_SIZE - 4)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.xlabel('Custom Score', fontsize=LABEL_SIZE + 4)
    plt.ylabel('Frequency', fontsize=LABEL_SIZE + 4)
    plt.xticks(fontsize=TICKS_SIZE + 4)
    plt.yticks(fontsize=TICKS_SIZE + 4)
    plt.savefig('hypergraph_data/analysis/pos_enr_histo.png', dpi=600)

    plt.show()


def plot_scatter():
    df = pd.read_csv('hypergraph_data/analysis/positive_edge_results-final.csv')

    plt.figure(figsize=(12, 8))
    plt.scatter(df['Range'], df['Count'])

    plt.xlabel('Range', fontsize=LABEL_SIZE + 4)
    plt.ylabel('Count', fontsize=LABEL_SIZE + 4)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks(fontsize=TICKS_SIZE)
    plt.yticks(fontsize=TICKS_SIZE)
    plt.savefig('hypergraph_data/analysis/pos_enr_range_count.png', dpi=600)
    plt.show()


def plot_combined_histograms():
    df = pd.read_csv('hypergraph_data/analysis/all_node_edge_weights-analysis.csv')
    filtered_df = df[(df['Label'] == 1) & (df['Correct'] == 1)]

    columns_to_plot = ['20', '21', '22']
    colors = ['blue', 'green', 'red']

    plt.figure(figsize=(10, 6))

    for col, color in zip(columns_to_plot, colors):
        data = pd.to_numeric(filtered_df[col], errors='coerce').dropna()

        plt.hist(data, bins=20, edgecolor=color, histtype='step', label=f'Column {col}', color=color)

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Combined Histograms for Columns 20, 21, 22')
    plt.legend()

    plt.show()


def plot_smoothed_histograms():
    df = pd.read_csv('hypergraph_data/analysis/all_node_edge_weights-analysis.csv')
    filtered_df = df[(df['Label'] == 1) & (df['Correct'] == 1)]

    columns_to_plot = ['20', '21', '22']
    labels = ['No APOE4 alleles', '1 APOE4 allele', '2 APOE3 alleles']

    plt.figure(figsize=(12, 8))

    cmap = plt.cm.viridis
    colors = [cmap(i) for i in np.linspace(0, 0.8, len(columns_to_plot))]

    for col, label, color in zip(columns_to_plot, labels, colors):
        data = pd.to_numeric(filtered_df[col], errors='coerce').dropna()

        sns.kdeplot(data, color=color, label=label)

    plt.xlabel('Edge score', fontsize=LABEL_SIZE + 4)
    plt.ylabel('Density', fontsize=LABEL_SIZE + 4)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks(fontsize=TICKS_SIZE)
    plt.yticks(fontsize=TICKS_SIZE)
    plt.legend(fontsize=LABEL_SIZE)
    plt.grid()
    plt.savefig('hypergraph_data/analysis/apoe4_pos.png')
    plt.show()

plot_smoothed_histograms()
plot_combined_histograms()
plot_scatter()
plot_custom_score_histogram()
plot_roc_comparison()
plot_aupr_comparison()
plot_roc()
plot_aupr()
plot_cohort_f1s()


df = pd.read_csv('hypergraph_data/analysis/3year-grid-worst.csv')
plot_loss_comparison(df, "worst", 200)
df = pd.read_csv('hypergraph_data/analysis/3year-grid-best.csv')
plot_loss_comparison(df, "best", 93)

