import pathlib

import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from matplotlib import patheffects

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

raw_data_filepath = pathlib.Path('hypergraph_data/raw_data.csv')

matplotlib.rcParams['font.family'] = 'Times New Roman'

TITLE_SIZE = 24
LABEL_SIZE = 24
TICKS_SIZE = 20
figx = 12
figy = 8
SIZE_ADJUST = 8

# Change only the cohort name to run the EDA for different cohorts
cohort_name = '4_visits_LESS_300_hg_input'

cohort_data = f'hypergraph_data/{cohort_name}/final_rows.csv'
cohort_path = f'hypergraph_data/eda/images/{cohort_name}/'


def get_cohort_progression_stats(folder_names, output_file):
    output = {'cohort': [], 'total': [], 'progressed': []}
    for folder in folder_names:
        output['cohort'].append(folder)
        df = pd.read_csv(f'hypergraph_data/{folder}/final_rows_with_outliers_and_md.csv')
        total = df.shape[0]
        prog = df[df.PROGRESSED == 1].shape[0]
        output['total'].append(total)
        output['progressed'].append(prog)
    df = pd.DataFrame(output)
    df.to_csv(output_file)
    return df


def generate_cohort_size_bar_chart():
    all_features_cohorts = [f"{prefix}_ALL_hg_input" for prefix in cohort_prefixes]
    all_features_size_analysis_output = 'hypergraph_data/eda/all_features_cohorts_analysis.csv'
    df = get_cohort_progression_stats(all_features_cohorts, all_features_size_analysis_output)

    df['year_label'] = [f"{i}" for i in range(1, len(df['cohort']) + 1)]

    cmap = cm.viridis_r
    norm = mcolors.Normalize(vmin=df['total'].min(), vmax=df['total'].max())
    colors = cmap(norm(df['total']))

    plt.figure(figsize=(figx, figy))
    bars = plt.bar(df['year_label'], df['total'], color=colors)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + max(df['total']) * 0.02, int(yval), va='bottom', ha='center',
                 fontsize=TICKS_SIZE + SIZE_ADJUST)

    plt.xlabel("Length of Participation (years)", fontsize=LABEL_SIZE + SIZE_ADJUST, labelpad=15)
    plt.ylabel("Number of Participants", fontsize=LABEL_SIZE + SIZE_ADJUST, labelpad=15)

    plt.xticks(fontsize=TICKS_SIZE + SIZE_ADJUST)
    plt.yticks(fontsize=TICKS_SIZE + SIZE_ADJUST)

    plt.ylim(0, max(df['total']) * 1.1)

    plt.tight_layout()
    plt.savefig('hypergraph_data/eda/images/cohort/number_by_length.png', dpi=600)
    plt.show()


def generate_cohort_progression_bar_chart():
    all_features_cohorts = [f"{prefix}_LESS_300_hg_input" for prefix in cohort_prefixes]
    all_features_size_analysis_output = 'hypergraph_data/eda/all_features_cohorts_analysis.csv'
    df = get_cohort_progression_stats(all_features_cohorts, all_features_size_analysis_output)

    df['progressed_percentage'] = (df['progressed'] / df['total']) * 100

    df['year_label'] = [f"{i}" for i in range(1, len(df['cohort']) + 1)]

    cmap = cm.viridis_r
    norm = mcolors.Normalize(vmin=df['progressed_percentage'].min(), vmax=df['progressed_percentage'].max())
    colors = cmap(norm(df['progressed_percentage']))

    plt.figure(figsize=(figx, figy))
    bars = plt.bar(df['year_label'], df['progressed_percentage'], color=colors)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.1f}%", va='bottom', ha='center', fontsize=TICKS_SIZE + SIZE_ADJUST)

    plt.xlabel("Length of Participation (years)", fontsize=LABEL_SIZE + SIZE_ADJUST, labelpad=15)
    plt.ylabel("Percentage Progressed", fontsize=LABEL_SIZE + SIZE_ADJUST, labelpad=15)

    plt.xticks(fontsize=TICKS_SIZE + SIZE_ADJUST)
    plt.yticks(fontsize=TICKS_SIZE + SIZE_ADJUST)
    plt.ylim(0, 45)

    plt.tight_layout()

    plt.savefig('hypergraph_data/eda/images/cohort/progression_percentage_by_length.png', dpi=600)
    plt.show()


def generate_all_cohort_age_box_plots():
    all_features_cohorts = [f"{prefix}_LESS_300_hg_input" for prefix in cohort_prefixes]

    data = []
    for folder in all_features_cohorts:
        df = pd.read_csv(f'hypergraph_data/{folder}/final_rows_with_outliers_and_md.csv')
        ages = df['NACCAGE']
        data.append(ages)

    labels = [f"{i}" for i in range(1,11)]

    plt.figure(figsize=(figx, figy))
    box = plt.boxplot(data, vert=True, patch_artist=True, labels=labels)

    colors = cm.viridis(np.linspace(0, 1, len(labels)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.xlabel('Length of Participation (years)', labelpad=15, fontsize=LABEL_SIZE + SIZE_ADJUST)
    plt.ylabel('Age', labelpad=15, fontsize=LABEL_SIZE + SIZE_ADJUST)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=TICKS_SIZE + SIZE_ADJUST)
    plt.yticks(fontsize=TICKS_SIZE + SIZE_ADJUST)

    plt.tight_layout()

    plt.savefig('hypergraph_data/eda/images/cohort/initial-ages-box-plots.png', dpi=600)
    plt.show()


def generate_missing_data_counts():
    all_features_cohorts = [f"{prefix}_LESS_300_hg_input" for prefix in cohort_prefixes]

    data_dict = {
        'name': ['1', '2', '3', '4', '5', '6', '7', '8', '9',
                 '10'],
        'total': [],
        'after_missing_removed': []
    }

    for folder in all_features_cohorts:
        df1 = pd.read_csv(f'hypergraph_data/{folder}/final_rows.csv')
        data_dict['after_missing_removed'].append(df1.shape[0])
        df2 = pd.read_csv(f'hypergraph_data/{folder}/final_rows_with_outliers_and_md.csv')
        data_dict['total'].append(df2.shape[0])

    df = pd.DataFrame(data_dict)

    df['missing_percentage'] = ((df['total'] - df['after_missing_removed']) / df['total']) * 100

    cmap = cm.viridis_r
    norm = mcolors.Normalize(vmin=df['missing_percentage'].min(), vmax=df['missing_percentage'].max())
    colors = cmap(norm(df['missing_percentage']))

    plt.figure(figsize=(figx, figy))
    bars = plt.bar(df['name'], df['missing_percentage'], color=colors)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.1f}%", va='bottom', ha='center',
                 fontsize=TICKS_SIZE)

    plt.xlabel("Length of Participation (years)", fontsize=LABEL_SIZE + SIZE_ADJUST, labelpad=15)
    plt.ylabel("Percentage of Missing Data", fontsize=LABEL_SIZE + SIZE_ADJUST, labelpad=15)

    plt.xticks(fontsize=TICKS_SIZE + SIZE_ADJUST)
    plt.yticks(fontsize=TICKS_SIZE + SIZE_ADJUST)

    plt.tight_layout()
    plt.ylim(0, 50)

    plt.savefig('hypergraph_data/eda/images/cohort/missing_data_percentage.png', dpi=600)
    plt.show()


def medical_feature_representation():
    df = pd.read_csv(cohort_data)
    df_filtered = df[medication_descriptions.keys()]
    medication_percentages = (df_filtered.sum() / len(df_filtered)) * 100

    medication_percentages_sorted = medication_percentages.sort_values()

    fig, ax = plt.subplots(figsize=(figx, figy))
    colors = cm.viridis_r(np.linspace(0, 1, len(medication_percentages_sorted)))
    bars = ax.barh(medication_percentages_sorted.index, medication_percentages_sorted, color=colors)

    for bar in bars:
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.1f}%", va='center',
                fontsize=TICKS_SIZE + SIZE_ADJUST)

    ax.set_yticks(np.arange(len(medication_percentages_sorted)))
    ax.set_yticklabels([medication_descriptions[code] for code in medication_percentages_sorted.index],
                       fontsize=TICKS_SIZE + SIZE_ADJUST)

    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])

    plt.savefig(f'{cohort_path}medication_percentage_horizontal.png', dpi=600)
    plt.show()


def comborbidity_feature_representation():
    df = pd.read_csv(cohort_data)

    comorbidities_percentages = {}

    for comorbidity, description in COMORBIDITIES_MAPPING.items():
        if comorbidity in ['HEARING', 'VISION']:
            percentage = (df[comorbidity] == 0).sum() / len(df) * 100
        else:
            percentage = ((df[comorbidity] == 1) | (df[comorbidity] == 2)).sum() / len(df) * 100
        comorbidities_percentages[description] = percentage

    comorbidities_percentages = pd.Series(comorbidities_percentages).sort_values()

    fig, ax = plt.subplots(figsize=(figx, figy))
    colors = cm.viridis_r(np.linspace(0, 1, len(comorbidities_percentages)))
    bars = ax.barh(comorbidities_percentages.index, comorbidities_percentages, color=colors)

    for bar in bars:
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.2f}%", va='center',
                fontsize=TICKS_SIZE)

    ax.set_yticks(np.arange(len(comorbidities_percentages)))
    ax.set_yticklabels(comorbidities_percentages.index, fontsize=TICKS_SIZE)
    ax.set_xticks([])

    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.savefig(f'{cohort_path}comborbidities_percentage_horizontal.png', dpi=600)
    plt.show()


def plot_gender_distribution():

    df = pd.read_csv(cohort_data)

    df['SEX'] = df['SEX'].map({1: 'Male', 2: 'Female'})

    gender_counts = df['SEX'].value_counts()

    cmap = cm.viridis_r
    colors = cmap(np.linspace(0.3, 0.7, len(gender_counts)))

    fig, ax = plt.subplots(figsize=(figx, figy))
    wedges, texts, autotexts = ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140,
                                      colors=colors)

    for text in texts:
        text.set_fontsize(LABEL_SIZE + SIZE_ADJUST)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
        autotext.set_fontsize(TICKS_SIZE + SIZE_ADJUST)
        autotext.set_fontname('sans serif')
        autotext.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground='black',)])

    plt.savefig(f'{cohort_path}gender_distribution.png', dpi=600)

    plt.tight_layout()

    plt.show()


def gender_progression_distribution():
    df = pd.read_csv(cohort_data)

    df['SEX'] = df['SEX'].map({1: 'Male', 2: 'Female'})
    df['PROGRESSED'] = df['PROGRESSED'].map({1: 'Progressed', 0: 'Not Progressed'})

    pivot_df = df.pivot_table(index='SEX', columns='PROGRESSED', aggfunc='size', fill_value=0)

    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(figx, figy))

    cmap = cm.viridis_r
    colors = cmap(np.linspace(0.3, 0.7, pivot_df.shape[1]))

    pivot_df.plot(kind='barh', stacked=True, color=colors, ax=ax)

    ax.legend(title="Status", fontsize=TICKS_SIZE + SIZE_ADJUST, title_fontsize= TICKS_SIZE + SIZE_ADJUST,  bbox_to_anchor=(0.75, -0.1), frameon=False, ncols=2)
    plt.xticks([])
    plt.yticks(fontsize=TICKS_SIZE + SIZE_ADJUST)

    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.1f%%', label_type='center', fontsize=TICKS_SIZE, font="sans serif",
                     color='white', weight='bold',
                     path_effects=[patheffects.withStroke(linewidth=2.5, foreground='black')])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(f'{cohort_path}gender_progression.png', dpi=600)

    plt.show()


def plot_race_distribution_percentage():
    df = pd.read_csv(cohort_data)

    race_mapping = {
        1: 'White',
        2: 'Black or African American',
        3: 'American Indian or Alaska Native',
        4: 'Native Hawaiian or Pacific Islander',
        5: 'Asian',
        6: 'Multiracial'
    }
    df['NACCNIHR'] = df['NACCNIHR'].map(race_mapping)

    race_counts = df['NACCNIHR'].value_counts(normalize=True) * 100

    race_counts = race_counts.sort_values(ascending=True)

    cmap = cm.viridis_r
    colors = cmap(np.linspace(0, 1, len(race_counts)))

    fig, ax = plt.subplots(figsize=(figx, figy))
    bars = ax.barh(race_counts.index, race_counts, color=colors)

    plt.xticks([])
    plt.yticks(fontsize=TICKS_SIZE + SIZE_ADJUST)

    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + ax.get_xlim()[1] * 0.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.2f}%', va='center', fontsize=LABEL_SIZE + SIZE_ADJUST)

    plt.xlim(0, 100)
    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.savefig(f'{cohort_path}race_distribution.png', dpi=600)

    plt.show()


def race_progression_distribution():
    df = pd.read_csv(cohort_data)

    race_mapping = {
        1: 'White',
        2: 'Black or African American',
        3: 'American Indian or Alaska Native',
        4: 'Native Hawaiian or Pacific Islander',
        5: 'Asian',
        6: 'Multiracial'
    }
    df['NACCNIHR'] = df['NACCNIHR'].map(race_mapping)
    df['PROGRESSED'] = df['PROGRESSED'].map({1: 'Progressed', 0: 'Not Progressed'})

    pivot_df = df.pivot_table(index='NACCNIHR', columns='PROGRESSED', aggfunc='size', fill_value=0)
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    order = [
        'Native Hawaiian or Pacific Islander',
        'American Indian or Alaska Native',
        'Asian',
        'Multiracial',
        'Black or African American',
        'White',
    ]

    pivot_df = pivot_df.reindex(order)
    fig, ax = plt.subplots(figsize=(figx, figy))

    cmap = cm.viridis_r
    colors = cmap(np.linspace(0.3, 0.7, pivot_df.shape[1]))

    pivot_df.plot(kind='barh', stacked=True, color=colors, ax=ax)

    ax.legend(title="Status", fontsize=TICKS_SIZE + SIZE_ADJUST, title_fontsize= TICKS_SIZE + SIZE_ADJUST,  bbox_to_anchor=(0.6, -0.1), frameon=False, ncols=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.yticks(fontsize=TICKS_SIZE + SIZE_ADJUST)


    for bars in ax.containers:
        for bar in bars:
            width = bar.get_width()
            if width != 0:  # Skip labeling 0.00%
                label = f'{width:.1f}%'
                x_position = bar.get_x() + width / 2
                ax.text(x_position, bar.get_y() + bar.get_height() / 2, label, ha='center', va='center',
                        fontsize=TICKS_SIZE, color='white', weight='heavy', font="sans serif",
                        path_effects=[patheffects.withStroke(linewidth=2.5, foreground='black')])

    plt.tight_layout()
    ax.set_xticks([])
    ax.set_ylabel('')
    plt.savefig(f'{cohort_path}race_progression_horizontal_chart.png', dpi=600)

    plt.show()


def plot_progression_age_boxplot():
    df = pd.read_csv('hypergraph_data/4_visits_LESS_300_hg_input/final_rows_with_outliers_and_md.csv')

    progressed_ages = df[df['PROGRESSED'] == 1]['NACCAGE']
    not_progressed_ages = df[df['PROGRESSED'] == 0]['NACCAGE']
    data = [not_progressed_ages, progressed_ages]

    labels = ['Not Progressed', 'Progressed']

    plt.figure(figsize=(figx, figy))
    box = plt.boxplot(data, vert=True, patch_artist=True, labels=labels)

    colors = cm.viridis(np.linspace(0.3, 0.7, len(labels)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.ylabel('Age', labelpad=15, fontsize=LABEL_SIZE)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=TICKS_SIZE)
    plt.yticks(fontsize=TICKS_SIZE)

    plt.tight_layout()

    plt.savefig(f'{cohort_path}progression-age-box-plots.png', dpi=600)
    plt.show()


def plot_education():
    df = pd.read_csv(cohort_data)
    bins = [0, 12, 16, 18, float('inf')]
    labels = ['Up to High School', 'Bachelors', 'Masters', 'Doctorate']
    df['XEDUGROUP'] = pd.cut(df['EDUC'], bins=bins, labels=labels, right=False)

    education_counts = df['XEDUGROUP'].value_counts().reindex(labels)

    cmap = cm.viridis_r
    colors = cmap(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(figx, figy))
    wedges, texts, autotexts = ax.pie(education_counts, labels=education_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)

    for text in texts:
        text.set_fontsize(LABEL_SIZE)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
        autotext.set_fontname('sans serif')
        autotext.set_fontsize(TICKS_SIZE)
        autotext.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground='black')])

    plt.tight_layout()
    plt.savefig(f'{cohort_path}education_distribution.png', dpi=600)
    plt.show()


def plot_education_progression():
    df = pd.read_csv(cohort_data)

    bins = [0, 12, 16, 18, float('inf')]
    labels = ['Up to High School', 'Bachelors', 'Masters', 'Doctorate']
    df['XEDUGROUP'] = pd.cut(df['EDUC'], bins=bins, labels=labels, right=False)
    df['PROGRESSED'] = df['PROGRESSED'].map({1: 'Progressed', 0: 'Not Progressed'})

    pivot_df = df.pivot_table(index='XEDUGROUP', columns='PROGRESSED', aggfunc='size', fill_value=0)
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(figx, figy))

    cmap = cm.viridis_r
    colors = cmap(np.linspace(0.3, 0.7, pivot_df.shape[1]))

    pivot_df.plot(kind='barh', stacked=True, color=colors, ax=ax)

    ax.set_ylabel("")
    ax.legend(title="Status", fontsize=TICKS_SIZE + SIZE_ADJUST, title_fontsize= TICKS_SIZE + SIZE_ADJUST,  bbox_to_anchor=(0.75, -0.1), frameon=False, ncols=2)
    plt.yticks(fontsize=TICKS_SIZE + SIZE_ADJUST)
    plt.xticks(fontsize=TICKS_SIZE + SIZE_ADJUST)

    for bars in ax.containers:
        for bar in bars:
            width = bar.get_width()
            if width != 0:  # Skip labeling 0.00%
                label = f'{width:.2f}%'
                x_position = bar.get_x() + width / 2
                ax.text(x_position, bar.get_y() + bar.get_height() / 2, label, ha='center', va='center',
                        fontsize=TICKS_SIZE, color='white', weight='bold', font="sans serif",
                        path_effects=[patheffects.withStroke(linewidth=2.5, foreground='black')])

    plt.tight_layout()
    ax.set_xticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.savefig(f'{cohort_path}education_progression_horizontal_chart.png', dpi=600)

    plt.show()


def plot_hearing_progression_comparison():
    df = pd.read_csv(cohort_data)

    df['HEARING'] = df['HEARING'].map({0: 'Hearing Loss', 1: 'No Hearing Loss'})
    df['HEARAID'] = df['HEARAID'].map({0: 'Does Not Wear Aid', 1: 'Wears Aid'})
    df['PROGRESSED'] = df['PROGRESSED'].map({1: 'Progressed', 0: 'Not Progressed'})

    conditions = [
        (df['HEARING'] == 'No Hearing Loss'),
        (df['HEARING'] == 'Hearing Loss') & (df['HEARAID'] == "Does Not Wear Aid"),
        (df['HEARING'] == 'Hearing Loss') & (df['HEARAID'] == "Wears Aid"),
    ]
    choices = ['No Hearing Loss', 'Hearing Loss, Does Not Wear Aid', 'Hearing Loss, Wears Aid']
    df['Category'] = np.select(conditions, choices)

    pivot_df = df.pivot_table(index='Category', columns='PROGRESSED', aggfunc='size', fill_value=0)
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = cm.viridis_r
    colors = cmap(np.linspace(0.3, 0.7, pivot_df.shape[1]))
    pivot_df.plot(kind='barh', stacked=True, color=colors, ax=ax)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    plt.yticks(fontsize=TICKS_SIZE + SIZE_ADJUST)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.legend(title="Status", fontsize=TICKS_SIZE + SIZE_ADJUST, title_fontsize= TICKS_SIZE + SIZE_ADJUST,  bbox_to_anchor=(0.6, -0.1), frameon=False, ncols=2)

    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.1f%%', label_type='center', fontsize=TICKS_SIZE + SIZE_ADJUST, color='white', weight='bold', font="sans serif",
                     path_effects=[patheffects.withStroke(linewidth=2.5, foreground='black')])

    plt.tight_layout()
    plt.savefig(f'{cohort_path}hearing_progression_comparison.png', dpi=600)
    plt.show()


def plot_beta_blockers_progression():
    df = pd.read_csv(cohort_data)

    df['NACCBETA'] = df['NACCBETA'].map({0: 'No', 1: 'Yes'})
    df['PROGRESSED'] = df['PROGRESSED'].map({1: 'Progressed', 0: 'Not Progressed'})

    pivot_df = df.pivot_table(index='NACCBETA', columns='PROGRESSED', aggfunc='size', fill_value=0)

    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(figx, figy))

    cmap = cm.viridis_r
    colors = cmap(np.linspace(0.3, 0.7, pivot_df.shape[1]))

    pivot_df.plot(kind='barh', stacked=True, color=colors, ax=ax)

    ax.set_ylabel("Uses Beta Blockers", fontsize=LABEL_SIZE + SIZE_ADJUST)
    ax.legend(title="Status", fontsize=TICKS_SIZE + SIZE_ADJUST, title_fontsize=TICKS_SIZE + SIZE_ADJUST, bbox_to_anchor=(0.75, -0.1), frameon=False, ncols=2)
    plt.xticks([])
    plt.yticks(fontsize=TICKS_SIZE + SIZE_ADJUST)

    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.2f%%', label_type='center', fontsize=TICKS_SIZE + SIZE_ADJUST,
                     color='white', weight='bold', font="sans serif",
                     path_effects=[patheffects.withStroke(linewidth=2.5, foreground='black')])
    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.savefig(f'{cohort_path}beta_blockers_progression.png', dpi=600)

    plt.show()


def plot_naccne4s_distribution():
    df = pd.read_csv(cohort_data)

    df['NACCNE4S'] = df['NACCNE4S'].map(NACCNE4S_correspondence_dict)

    naccne4s_counts = df['NACCNE4S'].value_counts()

    cmap = cm.viridis_r
    colors = cmap(np.linspace(0.3, 1, len(naccne4s_counts)))

    fig, ax = plt.subplots(figsize=(figx, figy))
    wedges, texts, autotexts = ax.pie(naccne4s_counts, labels=naccne4s_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)

    for text in texts:
        text.set_fontsize(LABEL_SIZE)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
        autotext.set_fontsize(TICKS_SIZE)
        autotext.set_fontname('sans serif')
        autotext.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground='black')])

    plt.tight_layout()
    plt.savefig(f'{cohort_path}naccne4s_distribution.png', dpi=600)

    plt.show()


def plot_naccne4s_progression():
    df = pd.read_csv(cohort_data)

    df['NACCNE4S'] = df['NACCNE4S'].map(NACCNE4S_correspondence_dict)

    df['PROGRESSED'] = df['PROGRESSED'].map({1: 'Progressed', 0: 'Not Progressed'})

    pivot_df = df.pivot_table(index='NACCNE4S', columns='PROGRESSED', aggfunc='size', fill_value=0)
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    order = ["2 copies of e4 allele", "1 copy of e4 allele", "No e4 allele"]
    pivot_df = pivot_df.loc[order]
    fig, ax = plt.subplots(figsize=(figx, figy))

    cmap = cm.viridis_r
    colors = cmap(np.linspace(0.3, 0.7, pivot_df.shape[1]))

    pivot_df.plot(kind='barh', stacked=True, color=colors, ax=ax)

    ax.set_ylabel("")
    ax.legend(title="Status", fontsize=TICKS_SIZE + SIZE_ADJUST, title_fontsize=TICKS_SIZE + SIZE_ADJUST, loc="lower right", bbox_to_anchor=(0.75, -0.1), frameon=False, ncols=2)
    plt.yticks(rotation=0, fontsize=TICKS_SIZE + SIZE_ADJUST)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.xticks([])

    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.2f%%', label_type='center', fontsize=TICKS_SIZE,
                     color='white', weight='bold', font="sans serif",
                     path_effects=[patheffects.withStroke(linewidth=2.5, foreground='black')])

    plt.tight_layout()
    plt.savefig(f'{cohort_path}naccne4s_progression.png', dpi=600)

    plt.show()


if not os.path.exists('hypergraph_data'):
    os.mkdir('hypergraph_data')

if not os.path.exists('hypergraph_data/eda'):
    os.mkdir('hypergraph_data/eda')
    os.mkdir('hypergraph_data/eda/images')
    os.mkdir('hypergraph_data/eda/images/cohort')
    os.mkdir(cohort_path)


generate_cohort_size_bar_chart()
generate_cohort_progression_bar_chart()
generate_all_cohort_age_box_plots()
generate_missing_data_counts()

medical_feature_representation()
comborbidity_feature_representation()

plot_gender_distribution()
gender_progression_distribution()

plot_race_distribution_percentage()
race_progression_distribution()

plot_education()
plot_education_progression()

plot_progression_age_boxplot()

plot_hearing_progression_comparison()

plot_beta_blockers_progression()

plot_naccne4s_distribution()
plot_naccne4s_progression()
