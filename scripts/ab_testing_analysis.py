import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

def load_data(file_path):
    
    df = pd.read_csv(file_path, delimiter='|', low_memory=False)
    return df

def clean_data(df):
    
    df_clean = df.dropna(subset=['TotalClaims', 'PostalCode', 'Province', 'Gender'])
    return df_clean

def perform_ttest(group_A, group_B, column):
    if len(group_A) == 0 or len(group_B) == 0:
        print("One of the groups is empty, cannot perform T-test.")
        return None
    t_stat, p_value = ttest_ind(group_A[column], group_B[column], equal_var=False)
    print(f'T-statistic: {t_stat}, P-value: {p_value}')
    return p_value

def perform_chi2_test(df, column_A, column_B):
    contingency_table = pd.crosstab(df[column_A], df[column_B])
    if contingency_table.size == 0:
        print("Contingency table is empty, cannot perform Chi2 test.")
        return None
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
    print(f'Chi2 Stat: {chi2_stat}, P-value: {p_val}')
    return p_val

def select_groups(df, column):
    unique_values = df[column].unique()
    if len(unique_values) < 2:
        print(f"Not enough unique values in {column} to create two groups.")
        return None, None
    return unique_values[0], unique_values[1]

def test_risk_by_province(df):
    print("Testing risk differences across provinces:")
    province_A, province_B = select_groups(df, 'Province')
    if province_A is None or province_B is None:
        return
    group_A = df[df['Province'] == province_A]
    group_B = df[df['Province'] == province_B]
    p_value = perform_ttest(group_A, group_B, 'TotalClaims')
    if p_value and p_value < 0.05:
        print(f"Reject the null hypothesis: There is a significant difference in risk between {province_A} and {province_B}.")
    else:
        print(f"Fail to reject the null hypothesis: No significant difference in risk between {province_A} and {province_B}.")

def test_risk_by_zipcode(df):
    print("Testing risk differences between zip codes:")
    zip_A, zip_B = select_groups(df, 'PostalCode')
    if zip_A is None or zip_B is None:
        return
    group_A = df[df['PostalCode'] == zip_A]
    group_B = df[df['PostalCode'] == zip_B]
    p_value = perform_ttest(group_A, group_B, 'TotalClaims')
    if p_value and p_value < 0.05:
        print(f"Reject the null hypothesis: There is a significant difference in risk between zip codes {zip_A} and {zip_B}.")
    else:
        print(f"Fail to reject the null hypothesis: No significant difference in risk between zip codes {zip_A} and {zip_B}.")

def test_profit_margin_by_zipcode(df):
    print("Testing profit margin differences between zip codes:")
    zip_A, zip_B = select_groups(df, 'PostalCode')
    if zip_A is None or zip_B is None:
        return
    if 'ProfitMargin' not in df.columns:
        df['ProfitMargin'] = df['TotalPremium'] - df['TotalClaims']
    group_A = df[df['PostalCode'] == zip_A]
    group_B = df[df['PostalCode'] == zip_B]
    p_value = perform_ttest(group_A, group_B, 'ProfitMargin')
    if p_value and p_value < 0.05:
        print(f"Reject the null hypothesis: There is a significant difference in profit margin between zip codes {zip_A} and {zip_B}.")
    else:
        print(f"Fail to reject the null hypothesis: No significant difference in profit margin between zip codes {zip_A} and {zip_B}.")

def test_risk_by_gender(df):
    print("Testing risk differences between women and men:")
    group_A = df[df['Gender'] == 'Male']
    group_B = df[df['Gender'] == 'Female']
    p_value = perform_chi2_test(df, 'Gender', 'TotalClaims')
    if p_value and p_value < 0.05:
        print("Reject the null hypothesis: There is a significant difference in risk between women and men.")
    else:
        print("Fail to reject the null hypothesis: No significant difference in risk between women and men.")
