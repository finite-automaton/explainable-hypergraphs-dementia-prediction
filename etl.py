import math
import os
import pathlib
import pandas as pd


# FILE PATHS
# Raw Data
raw_data_csv = pathlib.Path(f'hypergraph_data/raw_data/raw_data.csv')
raw_data_pickle = pathlib.Path('hypergraph_data/raw_data/raw_data.pkl')
codes_missing_values_FILEPATH = pathlib.Path('hypergraph_d/test_path/codes_missing_vals.csv')

# Group codes by type (number of vals)
ADMIN = ["NACCID", "NACCVNUM", "NACCAVST", "INRELY"]
HG_TWO_VALS = ["NACCNREX","STEPWISE","SOMATIC","EMOT","HXHYPER","PDOTHR","NACCAHTN","NACCHTNC","NACCACEI","NACCAAAS","NACCBETA","NACCCCBS","NACCDIUR","NACCVASD","NACCANGI","NACCLIPL","NACCNSD","NACCAC","NACCADEP","NACCAPSY","NACCAANX","NACCADMD","NACCPDMD","NACCEMD","NACCEPMD","NACCDBMD","DECCLIN","EYEMOVE","FOCLDEF","GAITDIS","EMPTY","DROPACT","BORED","HELPLESS","WRTHLESS","HOPELESS","AFRAID","BETTER","STAYHOME","MEMPROB","TOBAC30","DEP2YRS","DEPOTHR","NACCTBI","TOBAC100","DEL","HALL","AGIT","ELAT","DISN","DEPD","APA","MOT","ANX","IRR","APP","NITE","STROKE","HEARAID","VISCORR","HEARING","VISION","NACCFAM","NACCDAD","NACCMOM","PDNORMAL","SPIRITS","SATIS","ENERGY","WONDRFUL","HAPPY","DECIN","MOFALLS","BEDEP","BEAHALL","BEDEL","COGATTN","COGVIS","BEPERCH","MOGAIT","COGJUDG","BEVHALL","COGOTHR","COGLANG","MOTREM","BEDISIN","BEAPATHY","MOSLOW","BEIRRIT","BEAGIT","NACCFADM","NACCFFTD","PARK","BRNINJ","HYCEPH","DEP","OTHPSY","NACCADMU","NACCFTDM"]
HG_TWO_VALSX = ["ABRUPT", "HXSTROKE", "FOCLSYM", "FOCLSIGN"]
HG_THREE_VALS = ["HYPERTEN","THYROID","HYPERCHO","B12DEF","TRAUMCHR","TRAUMEXT","NCOTHR","TRAUMBRF","ABUSOTHR","PSYCDIS","ALCOHOL","CBSTROKE","CVAFIB","SEIZURES","CVOTHR","CBTIA","CVPACE","INCONTF","CVBYPASS","INCONTU","CVHATT","CVANGIO","DIABETES","CVCHF"]
HG_THREE_VALSX = ["DECSUB", "VISWCORR","HEARWAID"]
HG_FOUR_VALSX = ["DIABTYPE", "FRSTCHG"]
HG_FOUR_VALSY = ["INDEPEND", "RESIDENC"]
HG_FIVE_VALS = ["BEMODE", "COGMODE", "NACCMOTF", "MOMODE"]
HG_FIVE_VALSX = ["SHOPPING","TRAVEL","EVENTS","STOVE","REMDATES","MEALPREP","BILLS","PAYATTN","TAXES","GAMES"]
HG_FIVE_VALSY = ["NACCLIVS"]
HG_SIX_VALSX = ["BRADYKIN","TAPSRT","TAPSLF","HANDMOVR","TRESTFAC","TRESTRHD","TRESTLHD","TRESTRFT","TRESTLFT","RIGDNECK","RIGDUPRT","RIGDLORT","RIGDLOLF","TRACTRHD","TRACTLHD","HANDMOVL","HANDALTR","HANDALTL","LEGRT","LEGLF","FACEXP","POSTURE","SPEECH","ARISING","GAIT","POSSTAB","RIGDUPLF"]
HG_SIX_VALSXY = ["COURSE"]
HG_SIX_VALSY = ["MARISTAT", "NACCNIHR"]
HG_SEVEN_VALSX = ["PACKSPER"]
HG_NINE_VALS = ["NACCCOGF"]
HG_ELEVEN_VALS = ["NACCBEHF"]
HG_THIRTEEN_VALS = ["HACHIN"]
OTHERS = ["NACCBMI", "BPDIAS", "BPSYS", "HANDED", "EDUC", "NACCAGE", "NACCAPOE", "NACCNE4S", "SEX", "MEMORY", "ORIENT", "JUDGMENT", "COMMUN", "HOMEHOBB", "PERSCARE"]

# Data sets
FAMILY_AND_GENETICS = ["NACCFAM","NACCDAD","NACCMOM","NACCFADM","NACCFFTD", "NACCADMU","NACCFTDM", "NACCAPOE","NACCNE4S"]
COMORBODITIES = ["PARK","BRNINJ","HYCEPH","DEP","OTHPSY", "INCONTF","CVBYPASS","INCONTU","CVHATT","CVANGIO", "DIABETES", "CVCHF","ABUSOTHR","PSYCDIS", "ALCOHOL","PDOTHR", "CBSTROKE", "CVAFIB", "SEIZURES", "CVOTHR","THYROID","HYPERCHO", "NACCTBI","CBTIA", "HEARING", "HEARAID", "HEARWAID", "VISION", "B12DEF", "VISCORR", "VISWCORR"]
DEMO = ["SEX", "NACCAGE", "NACCNIHR", "HANDED", "EDUC"]
CDR = ["MEMORY","ORIENT","JUDGMENT","COMMUN","HOMEHOBB", "PERSCARE"]
SOCIAL = ["RESIDENC","NACCLIVS","MARISTAT"]
HEALTH = ["HYPERTEN", "TOBAC100", "PACKSPER"]
BEHAVIOUR_COGNITION = ["DECSUB", "INDEPEND", "COGLANG","MOTREM","BEDISIN","BEAPATHY","MOSLOW","MOGAIT","BEIRRIT","BEAGIT", "MOFALLS","COGJUDG", "BEDEP", "BEVHALL","BEAHALL","NACCCOGF", "BEDEL", "COGATTN", "NACCMOTF", "COGVIS", "BEPERCH", "NACCBEHF","MOMODE","FRSTCHG", "DECIN", "DEL","HALL","AGIT","ELAT","DISN","DEPD","APA","MOT","ANX","IRR","APP"]
MEDICATION = ["NACCAHTN","NACCHTNC","NACCACEI","NACCAAAS","NACCBETA","NACCCCBS","NACCDIUR","NACCVASD","NACCANGI","NACCLIPL","NACCNSD","NACCAC","NACCADEP","NACCAPSY","NACCAANX","NACCPDMD","NACCEMD","NACCEPMD","NACCDBMD"]
DEPENDENCE = ["SHOPPING","TRAVEL","EVENTS","STOVE","REMDATES","MEALPREP","BILLS","PAYATTN", "TAXES","GAMES"]
DEPRESSION = ["SPIRITS","EMPTY","DROPACT","BORED","HELPLESS","SATIS","WRTHLESS","ENERGY","HOPELESS","AFRAID","WONDRFUL","BETTER","HAPPY","STAYHOME","MEMPROB"]

LESS_300 = FAMILY_AND_GENETICS + COMORBODITIES + DEMO + CDR + SOCIAL + HEALTH + BEHAVIOUR_COGNITION + MEDICATION + DEPENDENCE + DEPRESSION
MORE_MISSING = ["BPSYS","BPDIAS","NACCBMI","DIABTYPE","DECCLIN","STROKE","CVPACE","TRAUMCHR","TRAUMEXT","NCOTHR","TRAUMBRF","STEPWISE","SOMATIC","EMOT","HXHYPER","HACHIN","ABRUPT","HXSTROKE","FOCLSYM","FOCLSIGN","PDNORMAL","BRADYKIN","TAPSRT","TAPSLF","HANDMOVR","TRESTFAC","TRESTRHD","TRESTLHD","TRESTRFT","TRESTLFT","RIGDNECK","RIGDUPRT","RIGDLORT","RIGDLOLF","TRACTRHD","TRACTLHD","HANDMOVL","HANDALTR","HANDALTL","LEGRT","LEGLF","FACEXP","POSTURE","SPEECH","ARISING","GAIT","POSSTAB","RIGDUPLF","EYEMOVE","FOCLDEF","GAITDIS"]


def generate_missing_counts(codes, df):
    missing_dict = {}
    for col in codes:
        val_counts = df[col].value_counts()
        val_counts = val_counts.transpose()
        col_codes = df[col].tolist()
        count = 0
        for code in col_codes:
            if not math.isnan(code) and code in val_counts:
                count += val_counts[code]
        missing_dict[col] = count
    missing = pd.DataFrame([missing_dict])
    missing.transpose().to_csv('data/missing_value_counts.csv')


def get_df_with_min_num_visits(df, num):
    """

    Parameters
    ----------
    df: Dataframe with NACCID column
    num: Minimum number of visits a participant should have to not be filtered out

    Returns
    -------
    A dataframe containing all rows for ids who have at least num visits and excluding others

    """
    df = df[df.NACCAVST >= num]
    return df


def add_second_CDR_column(df, tf):
    """

    Parameters
    ----------
    df: A dataframe containing NACCID and CDRGLOB columns, each NACCID should have
    more than one row associated with it.
    tf: How many visits after the first to measure CDR

    Returns
    -------
    The same dataframe with an additional XCDR2 column which represents the second CDRGLOB
    score in the sequence given. This should be used when the first CDRGLOB score is 0.5
    to assess whether they progressed or remained stable on their second visit.

    """

    # Function to get the CDRGLOB value for the second row of each NACCID
    def get_second_CDRGLOB_value(group):
        group['XCDR2'] = group.iloc[tf].CDRGLOB if len(group) > 1 else None
        return group

    # Apply the function to each group and create the CDR2 column
    return df.groupby('NACCID').apply(get_second_CDRGLOB_value).reset_index(drop=True)


def filter_to_first_row_per_id(df):
    """

    Parameters
    ----------
    df: dataframe with NACCID column and multiple rows per NACCID

    Returns
    -------
    A dataframe with only the first row for each NACCID. If being used to restrict data to e.g. first visit
    then ensure that the data is first sorted by NACCVNUM

    """

    return df.drop_duplicates(subset='NACCID').reset_index(drop=True)


def filter_valid_rows(df, tf):
    """

    Parameters
    ----------
    df: Dataframe with NACCID and CDRGLOB columns
    tf: number of visits (timeframe) that participant must have after and including having
    a GLOBCDR score of 0.5 to be included in result

    Returns
    -------
    A dataframe which includes only participants who have a row with a CDRGLOB of 0.5 and
    at least tf number of rows after that row. The dataframe includes only the row with the first occurrence
    of CDRGLOB == 0.5 and exactly tf subsequent rows for each NACCID, and excludes those who either have no
    CDRGLOB == 0.5 or do not have tf subsequent rows after their first row where CDRGLOB == 0.5
    """

    def filter_rows(group):
        # Find the index of the first instance of 0.5
        idx = group[group['CDRGLOB'].isin([0.5])].index.min()
        max_idx = group.index.max()

        # if no index is found then no CDR of 0.5 exists
        if idx is pd.NA:
            return None
        # return none if the first CDR of 0.5 index is also the last index
        if idx == max_idx:
            return None
        # if tf is 0 return all the rows after 0.5
        if tf == 0:
            return group.loc[idx:max_idx]

        # otherwise, return the CDR == 0.5 and subsequent tf rows, if not enough return None
        if 0 < tf <= (max_idx - idx + 1):
            return group.loc[idx:idx + (tf - 1)]
        else:
            return None

    # Filter the DataFrame based on the conditions
    filtered_groups = df.groupby('NACCID').apply(filter_rows).reset_index(drop=True)
    return filtered_groups


def replace_missing_values(all_rows, first_row, codes):
    # Create a copy of first_row df to store the results
    first_row_df = first_row.copy()

    # Iterate over each unique NACCID in first_row df
    for index, row in first_row.iterrows():
        naccid = row['NACCID']
        naccvnum = row['NACCVNUM']

        # Get the rows corresponding to the current NACCID in all_rows_df
        # and where NACCVNUM is less than the NACCVNUM in before_data
        rows = all_rows[(all_rows['NACCID'] == naccid) & (all_rows['NACCVNUM'] < naccvnum)]

        # Iterate over each column to replace missing values
        for column in first_row.columns:
            # Check if the column exists in the codes dictionary
            if column in codes:
                # Get the non-missing values for the current column and NACCID
                non_missing_values = rows[column][~rows[column].isin(codes[column])]

                # If there are any non-missing values, replace the value in before_data_df with the first non-missing value found
                if not non_missing_values.empty:
                    first_row_df.loc[index, column] = non_missing_values.iloc[0]

    return first_row_df.reset_index(drop=True)


def remove_missing_data(df, missing_codes):
    # Create a mask with all True values
    mask = pd.Series([True] * len(df))

    # Iterate over columns and update the mask to identify rows with missing values
    for column, missing_values in missing_codes.items():
        if column in df.columns:
            column_mask = ~df[column].isin(missing_values)
            mask = mask & column_mask

    # Use the mask to filter the DataFrame and remove rows with missing values
    filtered_data_df = df[mask].reset_index(drop=True)
    return filtered_data_df


def filter_data(selected_values, num_visits, processed_path, output_path):

    # Read raw data
    if not pathlib.Path.is_file(raw_data_pickle):
        raw_data = pd.read_csv(raw_data_csv)
        raw_data.to_pickle(raw_data_pickle)
    else:
        raw_data = pd.read_pickle(raw_data_pickle)

    # Filter to minimum num of visits
    if not pathlib.Path.is_file(pathlib.Path(f'{processed_path}/minimum_visits.pkl')):
        min_x_visits = get_df_with_min_num_visits(raw_data, num_visits)
        min_x_visits.to_pickle(pathlib.Path(f'{processed_path}/minimum_visits.pkl'))
    else:
        min_x_visits = pd.read_pickle(pathlib.Path(f'{processed_path}/minimum_visits.pkl'))
    # Filter to valid rows and add progressed column
    if not pathlib.Path.is_file(pathlib.Path(f'{processed_path}/valid_rows.pkl')):
        if not pathlib.Path.is_file(pathlib.Path(f'{processed_path}/valid_rows_pre.pkl')):
            valid_rows = filter_valid_rows(min_x_visits, num_visits)
            valid_rows.to_pickle(pathlib.Path(f'{processed_path}/valid_rows_pre.pkl'))
        else:
            valid_rows = pd.read_pickle(pathlib.Path(f'{processed_path}/valid_rows_pre.pkl'))
        # Add PROGRESSED column (label)
        valid_rows = add_second_CDR_column(valid_rows, num_visits - 1)
        valid_rows['PROGRESSED'] = valid_rows['XCDR2'] > 0.5
        valid_rows['PROGRESSED'] = valid_rows['PROGRESSED'].astype(int)
        valid_rows = valid_rows.drop(columns='XCDR2')
        # Filter to only the first 0.5 CDR visit
        valid_rows = filter_to_first_row_per_id(valid_rows)

        valid_rows.to_pickle(pathlib.Path(f'{processed_path}/valid_rows.pkl'))
    else:
        valid_rows = pd.read_pickle(pathlib.Path(f'{processed_path}/valid_rows.pkl'))

    # Remove any unreliable data
    valid_rows = valid_rows[valid_rows.INRELY == 0]

    # Filter to final cols
    final_cols = valid_rows.loc[:, selected_values + ['NACCID', 'NACCVNUM'] + ['PROGRESSED']].reset_index(drop=True)

    # save a version with missing data and age outliers
    final_cols.to_csv(f'{output_path}/final_rows_with_outliers_and_md.csv')

    # Remove age outliers
    if 'NACCAGE' in final_cols.columns:
        final_cols = final_cols[final_cols.NACCAGE >= 50].reset_index(drop=True)
        final_cols = final_cols[final_cols.NACCAGE <= 90].reset_index(drop=True)

    # remove rows with missing data
    missing_codes = pd.read_csv(pathlib.Path(f'hypergraph_data/raw_data/final_vals_missing_codes.csv'))

    final_data = remove_missing_data(final_cols, missing_codes)
    # no longer need to reference these columns for merging
    final_data = final_data.drop(columns=['NACCID', "NACCVNUM"])
    final_data.to_csv(f'{output_path}/final_rows.csv')
    return final_data


def associate_codes_with_funcs():
    def add_blood_pressure_value(df):
        df_copy = df.copy()
        df_copy['_BP_S2HT'] = 0
        df_copy['_BP_S1HT'] = 0
        df_copy['_BP_PHT'] = 0
        df_copy['_BP_NORM'] = 0
        for idx in df_copy.index:
            if df_copy['BPDIAS'][idx] > 99 or df_copy['BPSYS'][idx] > 159:
                df_copy['_BP_S2HT'][idx] = 1
            elif 90 <= df_copy['BPDIAS'][idx] < 99 or 139 <= df_copy['BPSYS'][idx] < 159:
                df_copy['_BP_S1HT'][idx] = 1
            elif 80 <= df_copy['BPDIAS'][idx] < 89 or 120 <= df_copy['BPSYS'][idx] < 139:
                df_copy['_BP_PHT'][idx] = 1
            else:
                df_copy['_BP_NORM'][idx] = 1

        return df_copy.reset_index(drop=True)

    def add_cdr_scores(df):
        df_copy = df.copy()
        cdr_cols = ["MEMORY", "ORIENT", "JUDGMENT", "COMMUN", "HOMEHOBB", "PERSCARE"]
        for col in cdr_cols:
            df_copy[f"_{col}_0"] = 0
            df_copy[f"_{col}_05"] = 0
            df_copy[f"_{col}_1"] = 0
            df_copy[f"_{col}_2"] = 0
            df_copy[f"_{col}_3"] = 0
            for idx in df_copy.index:
                if df_copy[col][idx] == 0.0:
                    df_copy[f"_{col}_0"][idx] = 1
                elif df_copy[col][idx] == 0.5:
                    df_copy[f"_{col}_05"][idx] = 1
                elif df_copy[col][idx] == 1.0:
                    df_copy[f"_{col}_1"][idx] = 1
                elif df_copy[col][idx] == 2.0:
                    df_copy[f"_{col}_2"][idx] = 1
                elif df_copy[col][idx] == 3.0:
                    df_copy[f"_{col}_3"][idx] = 1
        return df_copy.reset_index(drop=True)

    def add_education_value(df):
        df_copy = df.copy()
        df_copy["_EDU_LHS"] = 0
        df_copy["_EDU_HS"] = 0
        df_copy["_EDU_B"] = 0
        df_copy["_EDU_M"] = 0
        df_copy["_EDU_D"] = 0
        for idx in df_copy.index:
            if df_copy['EDUC'][idx] >= 20:
                df_copy['_EDU_D'][idx] = 1
            elif df_copy['EDUC'][idx] >= 18:
                df_copy['_EDU_M'][idx] = 1
            elif df_copy['EDUC'][idx] >= 16:
                df_copy['_EDU_B'][idx] = 1
            elif df_copy['EDUC'][idx] >= 12:
                df_copy["_EDU_HS"][idx] = 1
            else:
                df_copy["_EDU_LHS"][idx] = 1
        return df_copy.reset_index(drop=True)

    def add_age_value(df):
        df_copy = df.copy()
        min_age = df['NACCAGE'].min()
        max_age = df['NACCAGE'].max()
        # generate empty age rows
        smallest_bucket = min_age - (min_age % 5)
        largest_bucket = max_age + (5 - (max_age % 5))
        total_buckets = int((largest_bucket - smallest_bucket) / 5)
        start = smallest_bucket
        for bucket in range(0, total_buckets):
            df_copy[f'_AGE_{start}_{start + 5}'] = 0
            start = start + 5

        # populate rows
        for idx in df_copy.index:
            age = df_copy['NACCAGE'][idx]
            # get bucket for age
            age_bucket = age - (age % 5)
            df_copy[f'_AGE_{age_bucket}_{age_bucket + 5}'][idx] = 1

        return df_copy.reset_index(drop=True)

    def add_bmi_value(df):
        df_copy = df.copy()
        df_copy["_BMI_SUW"] = 0
        df_copy["_BMI_UW"] = 0
        df_copy["_BMI_NW"] = 0
        df_copy["_BMI_OW"] = 0
        df_copy["_BMI_OB"] = 0

        for idx in df_copy.index:
            bmi = df_copy['NACCBMI'][idx]
            if bmi < 16.5:
                df_copy['_BMI_SUW'][idx] = 1
            elif bmi < 18.5:
                df_copy['_BMI_UW'][idx] = 1
            elif bmi < 24.9:
                df_copy['_BMI_NW'][idx] = 1
            elif bmi < 29.9:
                df_copy["_BMI_OW"][idx] = 1
            else:
                df_copy["_BMI_OB"][idx] = 1

        return df_copy.reset_index(drop=True)

    def add_handed(df):
        df_copy = df.copy()
        df_copy['_HANDED_L'] = 0
        df_copy['_HANDED_R'] = 0
        df_copy['_HANDED_A'] = 0

        for idx in df_copy.index:
            handed = df_copy['HANDED'][idx]
            if handed == 1:
                df_copy['_HANDED_L'][idx] = 1
            if handed == 2:
                df_copy['_HANDED_R'][idx] = 1
            if handed == 3:
                df_copy['_HANDED_A'][idx] = 1
        return df_copy.reset_index(drop=True)

    def add_sex_value(df):
        df_copy = df.copy()
        df_copy['_MALE'] = 0
        df_copy['_FEMALE'] = 0
        for idx in df_copy.index:
            if df_copy['SEX'][idx] == 1:
                df_copy['_MALE'][idx] = 1
            if df_copy['SEX'][idx] == 2:
                df_copy['_FEMALE'][idx] = 1
        return df_copy.reset_index(drop=True)

    def add_apoe_genotype_vals(df):
        df_copy = df.copy()
        df_copy['_APOE_33'] = 0
        df_copy['_APOE_34'] = 0
        df_copy['_APOE_32'] = 0
        df_copy['_APOE_44'] = 0
        df_copy['_APOE_42'] = 0
        df_copy['_APOE_22'] = 0
        for idx in df_copy.index:
            apoe = df_copy['NACCAPOE'][idx]
            if apoe == 1:
                df_copy['_APOE_33'][idx] = 1
            if apoe == 2:
                df_copy['_APOE_34'][idx] = 1
            if apoe == 3:
                df_copy['_APOE_32'][idx] = 1
            if apoe == 4:
                df_copy['_APOE_44'][idx] = 1
            if apoe == 5:
                df_copy['_APOE_42'][idx] = 1
            if apoe == 6:
                df_copy['_APOE_22'][idx] = 1
        return df_copy.reset_index(drop=True)

    def add_e4_alleles(df):
        df_copy = df.copy()
        df_copy['_E4_0'] = 0
        df_copy['_E4_1'] = 0
        df_copy['_E4_2'] = 0
        for idx in df_copy.index:
            e4 = df_copy['NACCNE4S'][idx]
            if e4 == 0:
                df_copy['_E4_0'][idx] = 1
            if e4 == 1:
                df_copy['_E4_1'][idx] = 1
            if e4 == 2:
                df_copy['_E4_2'][idx] = 1
        return df_copy.reset_index(drop=True)

    def add_HG_TWO_VALS(df, col):
        df = df.copy()
        if col in HG_TWO_VALS:
            df[f'_{col}_0'] = 0
            df[f'_{col}_1'] = 0
            for idx in df.index:
                x = int(df[col][idx])
                df[f'_{col}_{x}'][idx] = 1
        return df.reset_index(drop=True)

    def add_HG_TWO_VALSX(df, col):
        df = df.copy()
        if col in HG_TWO_VALSX:
            df[f'_{col}_0'] = 0
            df[f'_{col}_2'] = 0
            for idx in df.index:
                x = int(df[col][idx])
                df[f'_{col}_{x}'][idx] = 1
        return df.reset_index(drop=True)

    def add_HG_THREE_VALS(df, col):
        df = df.copy()
        if col in HG_THREE_VALS:
            df[f'_{col}_0'] = 0
            df[f'_{col}_1'] = 0
            df[f'_{col}_2'] = 0
            for idx in df.index:
                x = int(df[col][idx])
                df[f'_{col}_{x}'][idx] = 1
        return df.reset_index(drop=True)

    def add_HG_THREE_VALSX(df, col):
        df = df.copy()
        if col in HG_THREE_VALSX:
            df[f'_{col}_0'] = 0
            df[f'_{col}_1'] = 0
            df[f'_{col}_8'] = 0
            for idx in df.index:
                x = int(df[col][idx])
                df[f'_{col}_{x}'][idx] = 1
        return df.reset_index(drop=True)

    def add_HG_FOUR_VALSY(df, col):
        df = df.copy()
        if col in HG_FOUR_VALSY:
            df[f'_{col}_1'] = 0
            df[f'_{col}_2'] = 0
            df[f'_{col}_3'] = 0
            df[f'_{col}_4'] = 0

            for idx in df.index:
                x = int(df[col][idx])
                df[f'_{col}_{x}'][idx] = 1
        return df.reset_index(drop=True)

    def add_HG_FOUR_VALSX(df, col):
        df = df.copy()
        if col in HG_FOUR_VALSX:
            df[f'_{col}_1'] = 0
            df[f'_{col}_2'] = 0
            df[f'_{col}_3'] = 0
            df[f'_{col}_8'] = 0

            for idx in df.index:
                x = df[col][idx]
                df[f'_{col}_{x}'][idx] = 1
        return df.reset_index(drop=True)

    def add_HG_FIVE_VALS(df, col):
        df = df.copy()
        if col in HG_FIVE_VALS:
            df[f'_{col}_0'] = 0
            df[f'_{col}_1'] = 0
            df[f'_{col}_2'] = 0
            df[f'_{col}_3'] = 0
            df[f'_{col}_4'] = 0

            for idx in df.index:
                x = int(df[col][idx])
                df[f'_{col}_{x}'][idx] = 1
        return df.reset_index(drop=True)

    def add_HG_FIVE_VALSX(df, col):
        df = df.copy()
        if col in HG_FIVE_VALSX:
            df[f'_{col}_0'] = 0
            df[f'_{col}_1'] = 0
            df[f'_{col}_2'] = 0
            df[f'_{col}_3'] = 0
            df[f'_{col}_8'] = 0

            for idx in df.index:
                x = int(df[col][idx])
                df[f'_{col}_{x}'][idx] = 1
        return df.reset_index(drop=True)

    def add_HG_FIVE_VALSY(df, col):
        df = df.copy()
        if col in HG_FIVE_VALSY:
            df[f'_{col}_1'] = 0
            df[f'_{col}_2'] = 0
            df[f'_{col}_3'] = 0
            df[f'_{col}_4'] = 0
            df[f'_{col}_5'] = 0

            for idx in df.index:
                x = int(df[col][idx])
                df[f'_{col}_{x}'][idx] = 1
        return df.reset_index(drop=True)

    def add_HG_SIX_VALSX(df, col):
        df = df.copy()
        if col in HG_SIX_VALSX:
            df[f'_{col}_0'] = 0
            df[f'_{col}_1'] = 0
            df[f'_{col}_2'] = 0
            df[f'_{col}_3'] = 0
            df[f'_{col}_4'] = 0
            df[f'_{col}_8'] = 0

            for idx in df.index:
                x = int(df[col][idx])
                df[f'_{col}_{x}'][idx] = 1
        return df.reset_index(drop=True)

    def add_HG_SIX_VALSY(df, col):
        df = df.copy()
        if col in HG_SIX_VALSY:
            df[f'_{col}_1'] = 0
            df[f'_{col}_2'] = 0
            df[f'_{col}_3'] = 0
            df[f'_{col}_4'] = 0
            df[f'_{col}_5'] = 0
            df[f'_{col}_6'] = 0

            for idx in df.index:
                x = int(df[col][idx])
                df[f'_{col}_{x}'][idx] = 1
        return df.reset_index(drop=True)

    def add_HG_SIX_VALSXY(df, col):
        df = df.copy()
        if col in HG_SIX_VALSXY:
            df[f'_{col}_1'] = 0
            df[f'_{col}_2'] = 0
            df[f'_{col}_3'] = 0
            df[f'_{col}_4'] = 0
            df[f'_{col}_5'] = 0
            df[f'_{col}_8'] = 0

            for idx in df.index:
                x = int(df[col][idx])
                df[f'_{col}_{x}'][idx] = 1
        return df.reset_index(drop=True)

    def add_HG_SEVEN_VALSX(df, col):
        df = df.copy()
        if col in HG_SEVEN_VALSX:
            df[f'_{col}_0'] = 0
            df[f'_{col}_1'] = 0
            df[f'_{col}_2'] = 0
            df[f'_{col}_3'] = 0
            df[f'_{col}_4'] = 0
            df[f'_{col}_5'] = 0
            df[f'_{col}_8'] = 0

            for idx in df.index:
                x = int(df[col][idx])
                df[f'_{col}_{x}'][idx] = 1
        return df.reset_index(drop=True)

    def add_HG_NINE_VALS(df, col):
        df = df.copy()
        if col in HG_NINE_VALS:
            df[f'_{col}_0'] = 0
            df[f'_{col}_1'] = 0
            df[f'_{col}_2'] = 0
            df[f'_{col}_3'] = 0
            df[f'_{col}_4'] = 0
            df[f'_{col}_5'] = 0
            df[f'_{col}_6'] = 0
            df[f'_{col}_7'] = 0
            df[f'_{col}_8'] = 0

            for idx in df.index:
                x = int(df[col][idx])
                df[f'_{col}_{x}'][idx] = 1
        return df.reset_index(drop=True)

    def add_HG_ELEVEN_VALS(df, col):
        df = df.copy()
        if col in HG_ELEVEN_VALS:
            df[f'_{col}_0'] = 0
            df[f'_{col}_1'] = 0
            df[f'_{col}_2'] = 0
            df[f'_{col}_3'] = 0
            df[f'_{col}_4'] = 0
            df[f'_{col}_5'] = 0
            df[f'_{col}_6'] = 0
            df[f'_{col}_7'] = 0
            df[f'_{col}_8'] = 0
            df[f'_{col}_9'] = 0
            df[f'_{col}_10'] = 0
            df[f'_{col}_11'] = 0

            for idx in df.index:
                x = int(df[col][idx])
                df[f'_{col}_{x}'][idx] = 1
        return df.reset_index(drop=True)

    def add_HG_THIRTEEN_VALS(df, col):
        df = df.copy()
        if col in HG_THIRTEEN_VALS:
            df[f'_{col}_0'] = 0
            df[f'_{col}_1'] = 0
            df[f'_{col}_2'] = 0
            df[f'_{col}_3'] = 0
            df[f'_{col}_4'] = 0
            df[f'_{col}_5'] = 0
            df[f'_{col}_6'] = 0
            df[f'_{col}_7'] = 0
            df[f'_{col}_8'] = 0
            df[f'_{col}_9'] = 0
            df[f'_{col}_10'] = 0
            df[f'_{col}_11'] = 0
            df[f'_{col}_12'] = 0

            for idx in df.index:
                x = int(df[col][idx])
                df[f'_{col}_{x}'][idx] = 1
        return df.reset_index(drop=True)

    code_type_dict = {}

    def add_to_code_type_dict(list, function):
        for code in list:
            code_type_dict[code] = function

    add_to_code_type_dict(HG_TWO_VALS, add_HG_TWO_VALS)
    add_to_code_type_dict(HG_TWO_VALSX, add_HG_TWO_VALSX)
    add_to_code_type_dict(HG_THREE_VALS, add_HG_THREE_VALS)
    add_to_code_type_dict(HG_THREE_VALSX, add_HG_THREE_VALSX)
    add_to_code_type_dict(HG_FOUR_VALSX, add_HG_FOUR_VALSX)
    add_to_code_type_dict(HG_FOUR_VALSY, add_HG_FOUR_VALSY)
    add_to_code_type_dict(HG_FIVE_VALS, add_HG_FIVE_VALS)
    add_to_code_type_dict(HG_FIVE_VALSX, add_HG_FIVE_VALSX)
    add_to_code_type_dict(HG_FIVE_VALSY, add_HG_FIVE_VALSY)
    add_to_code_type_dict(HG_SIX_VALSX, add_HG_SIX_VALSX)
    add_to_code_type_dict(HG_SIX_VALSXY, add_HG_SIX_VALSXY)
    add_to_code_type_dict(HG_SIX_VALSY, add_HG_SIX_VALSY)
    add_to_code_type_dict(HG_SEVEN_VALSX, add_HG_SEVEN_VALSX)
    add_to_code_type_dict(HG_NINE_VALS, add_HG_NINE_VALS)
    add_to_code_type_dict(HG_ELEVEN_VALS, add_HG_ELEVEN_VALS)
    add_to_code_type_dict(HG_THIRTEEN_VALS, add_HG_THIRTEEN_VALS)
    add_to_code_type_dict(["NACCBMI"], add_bmi_value)
    add_to_code_type_dict(["BPDIAS", "BPSYS"], add_blood_pressure_value)
    add_to_code_type_dict(["HANDED"], add_handed)
    add_to_code_type_dict(["EDUC"], add_education_value)
    add_to_code_type_dict(["NACCAGE"], add_age_value)
    add_to_code_type_dict(["NACCAPOE"], add_apoe_genotype_vals)
    add_to_code_type_dict(["NACCNE4S"], add_e4_alleles)
    add_to_code_type_dict(["SEX"], add_sex_value)
    add_to_code_type_dict(["MEMORY", "ORIENT", "JUDGMENT", "COMMUN", "HOMEHOBB", "PERSCARE"], add_cdr_scores)

    return code_type_dict


def generate_hypergraph_vals(df, features, path):

    code_type_dict = associate_codes_with_funcs()
    hg = df.copy()

    for col in df:
        if col not in features:
            continue
        if col == "BPSYS":
            continue
        if col in OTHERS:
            hg = code_type_dict[col](hg)
        else:
            hg = code_type_dict[col](hg, col)

    final_hypergraph_df = hg[[col for col in hg if col.startswith('_')]].reset_index(drop=True)
    # drop empty edges
    for col in final_hypergraph_df:
        if final_hypergraph_df[col].sum() == 0:
            final_hypergraph_df = final_hypergraph_df.drop(columns=col)
    final_hypergraph_df.to_csv(f'{path}/final_hypergraph.csv')
    return final_hypergraph_df


def add_progressed_vals(df):
    df_copy = df.copy()
    df_copy['PROGRESSED'] = 0
    for idx in df_copy.index:
        if df_copy['XCDR2'][idx] > 0.5:
            df_copy['PROGRESSED'][idx] = 1
    return df_copy.reset_index(drop=True)


def transform_df_to_content_file(df, H, path):
    num_edges = H.shape[1] - 1
    existing_rows = len(df)
    for x in range(num_edges + 1):
        df.loc[existing_rows + x] = 0
    # return
    # add empty feature rows for edges
    df.to_csv(f'{path}/nacc.content', sep=' ', index=True, header=False)


def transform_H_to_edges(H, path):
    num_cols, _ = H.shape
    edge_start_index = num_cols  # add 1 more to account for header and start
    edge_mapping = {edge_name: i for i, edge_name in enumerate(H.columns, start=edge_start_index)}

    with open(f'{path}/nacc.edges', 'w') as f:
        for col in H.columns:
            nodes_connected = H.index[H[col] == 1].tolist()
            for node in nodes_connected:
                f.write(f"{node}\t{edge_mapping[col]}\n")

    # record a reference of edge index to name
    with open(f'{path}/edge_reference.csv', 'w') as f:
        start = -1
        for col in H.columns:
            if start == -1:  # skip the index
                start += 1
                continue
            f.write(f"{start}, {col}\n")
            start += 1


def create_hypergraph_files(features, num_visits, features_name):
    root = 'hypergraph_data'
    visits_path = f'{num_visits}_visits'
    processed_path = f'{root}/{visits_path}/processed_data'
    hg_input_path = f'{root}/{visits_path}_{features_name}_hg_input'

    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    if not os.path.exists(hg_input_path):
        os.makedirs(hg_input_path)

    data = filter_data(features, num_visits, processed_path, hg_input_path)
    H = generate_hypergraph_vals(data, features, hg_input_path)
    transform_H_to_edges(H, hg_input_path)
    transform_df_to_content_file(data, H, hg_input_path)


def generate_first_cohorts():
    create_hypergraph_files(LESS_300, 2, 'LESS_300')
    create_hypergraph_files(LESS_300, 3, 'LESS_300')
    create_hypergraph_files(LESS_300, 4, 'LESS_300')
    create_hypergraph_files(LESS_300, 5, 'LESS_300')
    create_hypergraph_files(LESS_300, 6, 'LESS_300')
    create_hypergraph_files(LESS_300, 7, 'LESS_300')
    create_hypergraph_files(LESS_300, 8, 'LESS_300')
    create_hypergraph_files(LESS_300, 9, 'LESS_300')
    create_hypergraph_files(LESS_300, 10, 'LESS_300')
    create_hypergraph_files(LESS_300, 11, 'LESS_300')

    create_hypergraph_files(LESS_300 + MORE_MISSING, 2, 'ALL')
    create_hypergraph_files(LESS_300 + MORE_MISSING, 3, 'ALL')
    create_hypergraph_files(LESS_300 + MORE_MISSING, 4, 'ALL')
    create_hypergraph_files(LESS_300 + MORE_MISSING, 5, 'ALL')
    create_hypergraph_files(LESS_300 + MORE_MISSING, 6, 'ALL')
    create_hypergraph_files(LESS_300 + MORE_MISSING, 7, 'ALL')
    create_hypergraph_files(LESS_300 + MORE_MISSING, 8, 'ALL')
    create_hypergraph_files(LESS_300 + MORE_MISSING, 9, 'ALL')
    create_hypergraph_files(LESS_300 + MORE_MISSING, 10, 'ALL')
    create_hypergraph_files(LESS_300 + MORE_MISSING, 11, 'ALL')


    create_hypergraph_files(FAMILY_AND_GENETICS, 2, 'FAMILY_AND_GENETICS')
    create_hypergraph_files(FAMILY_AND_GENETICS, 3, 'FAMILY_AND_GENETICS')
    create_hypergraph_files(FAMILY_AND_GENETICS, 4, 'FAMILY_AND_GENETICS')
    create_hypergraph_files(FAMILY_AND_GENETICS, 5, 'FAMILY_AND_GENETICS')
    create_hypergraph_files(FAMILY_AND_GENETICS, 6, 'FAMILY_AND_GENETICS')
    create_hypergraph_files(FAMILY_AND_GENETICS, 7, 'FAMILY_AND_GENETICS')
    create_hypergraph_files(FAMILY_AND_GENETICS, 8, 'FAMILY_AND_GENETICS')
    create_hypergraph_files(FAMILY_AND_GENETICS, 9, 'FAMILY_AND_GENETICS')
    create_hypergraph_files(FAMILY_AND_GENETICS, 10, 'FAMILY_AND_GENETICS')
    create_hypergraph_files(FAMILY_AND_GENETICS, 11, 'FAMILY_AND_GENETICS')

    create_hypergraph_files(COMORBODITIES, 2, 'COMORBODITIES')
    create_hypergraph_files(COMORBODITIES, 3, 'COMORBODITIES')
    create_hypergraph_files(COMORBODITIES, 4, 'COMORBODITIES')
    create_hypergraph_files(COMORBODITIES, 5, 'COMORBODITIES')
    create_hypergraph_files(COMORBODITIES, 6, 'COMORBODITIES')
    create_hypergraph_files(COMORBODITIES, 7, 'COMORBODITIES')
    create_hypergraph_files(COMORBODITIES, 8, 'COMORBODITIES')
    create_hypergraph_files(COMORBODITIES, 9, 'COMORBODITIES')
    create_hypergraph_files(COMORBODITIES, 10, 'COMORBODITIES')
    create_hypergraph_files(COMORBODITIES, 11, 'COMORBODITIES')

    create_hypergraph_files(MEDICATION, 2, 'MEDICATION')
    create_hypergraph_files(MEDICATION, 3, 'MEDICATION')
    create_hypergraph_files(MEDICATION, 4, 'MEDICATION')
    create_hypergraph_files(MEDICATION, 5, 'MEDICATION')
    create_hypergraph_files(MEDICATION, 6, 'MEDICATION')
    create_hypergraph_files(MEDICATION, 7, 'MEDICATION')
    create_hypergraph_files(MEDICATION, 8, 'MEDICATION')
    create_hypergraph_files(MEDICATION, 9, 'MEDICATION')
    create_hypergraph_files(MEDICATION, 10, 'MEDICATION')
    create_hypergraph_files(MEDICATION, 11, 'MEDICATION')

    create_hypergraph_files(CDR, 2, 'CDR')
    create_hypergraph_files(CDR, 3, 'CDR')
    create_hypergraph_files(CDR, 4, 'CDR')
    create_hypergraph_files(CDR, 5, 'CDR')
    create_hypergraph_files(CDR, 6, 'CDR')
    create_hypergraph_files(CDR, 7, 'CDR')
    create_hypergraph_files(CDR, 8, 'CDR')
    create_hypergraph_files(CDR, 9, 'CDR')
    create_hypergraph_files(CDR, 10, 'CDR')
    create_hypergraph_files(CDR, 11, 'CDR')

    create_hypergraph_files(DEPENDENCE, 2, 'DEPENDENCE')
    create_hypergraph_files(DEPENDENCE, 3, 'DEPENDENCE')
    create_hypergraph_files(DEPENDENCE, 4, 'DEPENDENCE')
    create_hypergraph_files(DEPENDENCE, 5, 'DEPENDENCE')
    create_hypergraph_files(DEPENDENCE, 6, 'DEPENDENCE')
    create_hypergraph_files(DEPENDENCE, 7, 'DEPENDENCE')
    create_hypergraph_files(DEPENDENCE, 8, 'DEPENDENCE')
    create_hypergraph_files(DEPENDENCE, 9, 'DEPENDENCE')
    create_hypergraph_files(DEPENDENCE, 10, 'DEPENDENCE')
    create_hypergraph_files(DEPENDENCE, 11, 'DEPENDENCE')


def generate_ablations():
    ABL_FAMILY_AND_GENETICS = [x for x in LESS_300 if x not in FAMILY_AND_GENETICS]
    ABL_COMORBIDITIES = [x for x in LESS_300 if x not in COMORBODITIES]
    ABL_DEMO = [x for x in LESS_300 if x not in DEMO]
    ABL_CDR = [x for x in LESS_300 if x not in CDR]
    ABL_SOCIAL = [x for x in LESS_300 if x not in SOCIAL]
    ABL_HEALTH = [x for x in LESS_300 if x not in HEALTH]
    ABL_BEHAVIOUR_COGNITION = [x for x in LESS_300 if x not in BEHAVIOUR_COGNITION]
    ABL_MEDICATION = [x for x in LESS_300 if x not in MEDICATION]
    ABL_DEPENDENCE = [x for x in LESS_300 if x not in DEPENDENCE]
    ABL_DEPRESSION = [x for x in LESS_300 if x not in DEPRESSION]

    create_hypergraph_files(ABL_FAMILY_AND_GENETICS, 4, 'ABL_FAMILY_AND_GENETICS')
    create_hypergraph_files(ABL_COMORBIDITIES, 4, 'ABL_COMORBIDITIES')
    create_hypergraph_files(ABL_DEMO, 4, 'ABL_DEMO')
    create_hypergraph_files(ABL_CDR, 4, 'ABL_CDR')

    create_hypergraph_files(ABL_SOCIAL, 4, 'ABL_SOCIAL')
    create_hypergraph_files(ABL_HEALTH, 4, 'ABL_HEALTH')
    create_hypergraph_files(ABL_BEHAVIOUR_COGNITION, 4, 'ABL_BEHAVIOUR_COGNITION')
    create_hypergraph_files(ABL_MEDICATION, 4, 'ABL_MEDICATION')
    create_hypergraph_files(ABL_DEPENDENCE, 4, 'ABL_DEPENDENCE')
    create_hypergraph_files(ABL_DEPRESSION, 4, 'ABL_DEPRESSION')


def generate_final_set():
    FINAL_SET_2 = [x for x in LESS_300 if x not in MEDICATION]
    FINAL_SET = [x for x in FINAL_SET_2 if x not in DEPRESSION]

    create_hypergraph_files(FINAL_SET, 4, 'FINAL_SET_WITHGEN')


generate_first_cohorts()
