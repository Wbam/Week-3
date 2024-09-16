import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer

def load_data(file_path, delimiter='|'):
    
    data = pd.read_csv(file_path, delimiter=delimiter, low_memory=False)
    return data

def print_data_types(data):
   
    print("\nData Types of Each Column:")
    print(data.dtypes)

def print_descriptive_statistics(data):
    print("\nDescriptive Statistics for Numerical Features:")
    print(data.describe(include=[float, int]))

def print_additional_variability_measures(data, numerical_features):
    print("\nAdditional Variability Measures:")
    for feature in numerical_features:
        if feature in data.columns:
            print(f"\nStatistics for {feature}:")
            print(f"Variance: {data[feature].var()}")
            print(f"Standard Deviation: {data[feature].std()}")
            print(f"Mean: {data[feature].mean()}")
            print(f"Median: {data[feature].median()}")
            print(f"Min: {data[feature].min()}")
            print(f"Max: {data[feature].max()}")
def print_data_summary(data):
    
    print("\nData Summary:")
    print(data.describe(include='all'))

def print_numerical_descriptive_statistics(data, numerical_features):
    
    print("\nDescriptive Statistics for Numerical Features:")
    print(data[numerical_features].describe())

    print("\nVariability for Numerical Features:")
    print(data[numerical_features].std())
    
def describe_data(df):
    return df.describe(include='all', datetime_is_numeric=True)

def convert_dates(data, date_columns):
    for column, date_format in date_columns.items():
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors='coerce', format=date_format)
    return data

def convert_categorical(data, categorical_features):
    
    for feature in categorical_features:
        if feature in data.columns:
            data[feature] = data[feature].astype('category')
    return data

def convert_to_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace(',', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Column '{col}' not found in DataFrame.")
    return df

def handle_categorical_data(df, categorical_features):

    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].fillna('Unknown')
        else:
            print(f"Column '{col}' not found in DataFrame.")
    return df

def impute_numerical_data(df, numerical_features):
    imputer_num = SimpleImputer(strategy='median')
    df[numerical_features] = imputer_num.fit_transform(df[numerical_features])
    return df

def drop_columns_with_missing_data(df, columns_to_drop):
    
    df = df.drop(columns=columns_to_drop)
    return df

def plot_numerical_distributions(data, numerical_features):
    
    # Check if columns are numeric and filter out non-numeric ones
    numerical_features = [col for col in numerical_features if pd.api.types.is_numeric_dtype(data[col])]
    
    num_features = len(numerical_features)
    num_cols = 4  # Number of columns in the grid
    num_rows = int(np.ceil(num_features / num_cols))  # Calculate the number of rows needed
    
    plt.figure(figsize=(num_cols * 4, num_rows * 4))
    sns.set(style="whitegrid")
    
    # Plot histograms with KDE for numerical columns
    for i, col in enumerate(numerical_features, 1):
        plt.subplot(num_rows, num_cols, i)
        if col in data.columns:
            # Ensure the column data is numeric
            data[col] = pd.to_numeric(data[col], errors='coerce') 
            sns.histplot(data[col].dropna(), bins=30, color='skyblue', kde=True, kde_kws={'bw_adjust': 0.5})
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        else:
            plt.title(f'Column {col} not found')
            plt.xlabel('N/A')
            plt.ylabel('N/A')
    
    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(data, categorical_features):
    num_features = len(categorical_features)
    num_cols = 3  # Number of columns in the grid
    num_rows = int(np.ceil(num_features / num_cols))  # Calculate the number of rows needed

    plt.figure(figsize=(num_cols * 5, num_rows * 5))  # Increase figure size

    for i, col in enumerate(categorical_features, 1):
        plt.subplot(num_rows, num_cols, i)
        if col in data.columns:
            data[col].value_counts().plot(kind='bar', color='skyblue')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        else:
            plt.title(f'Column {col} not found')
            plt.xlabel('N/A')
            plt.ylabel('N/A')

    plt.subplots_adjust(hspace=1.2, wspace=0.2)  
    plt.show()

def plot_totalpremium_vs_totalclaims(data):
    
    data['TotalPremium'] = pd.to_numeric(data['TotalPremium'], errors='coerce')
    data['TotalClaims'] = pd.to_numeric(data['TotalClaims'], errors='coerce')

    if data['PostalCode'].dtype == 'int64':
        data['PostalCode'] = data['PostalCode'].astype('category')

    top_postal_codes = data['PostalCode'].value_counts().nlargest(10).index
    data_reduced = data[data['PostalCode'].isin(top_postal_codes)]

    # Plot
    plt.figure(figsize=(15, 10))
    sns.scatterplot(data=data_reduced, x='TotalPremium', y='TotalClaims', hue='PostalCode', palette='viridis', alpha=0.7)

    plt.title('TotalPremium vs. TotalClaims by PostalCode')
    plt.xlabel('TotalPremium')
    plt.ylabel('TotalClaims')

    plt.legend(title='PostalCode', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=8)  
    plt.subplots_adjust(right=0.8)
    plt.show()

def plot_correlation_matrix(data):
    
    data['TotalPremium'] = pd.to_numeric(data['TotalPremium'], errors='coerce')
    data['TotalClaims'] = pd.to_numeric(data['TotalClaims'], errors='coerce')

    # Calculate the correlation matrix
    correlation_matrix = data[['TotalPremium', 'TotalClaims']].corr()

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of TotalPremium and TotalClaims')
    plt.tight_layout()
    plt.show()

def plot_cover_type_by_province(data):
    # Convert 'CoverType' and 'Province' to categorical data types
    data['CoverType'] = data['CoverType'].astype('category')
    data['Province'] = data['Province'].astype('category')

    # Create the count plot
    plt.figure(figsize=(15, 10))
    sns.countplot(data=data, x='Province', hue='CoverType', palette='viridis')
    plt.title('Distribution of Insurance Cover Type Across Provinces')
    plt.xlabel('Province')
    plt.ylabel('Count')
    plt.xticks(rotation=45)  
    plt.legend(title='Cover Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_totalpremium_by_province_and_covercategory(data):

    data['TotalPremium'] = pd.to_numeric(data['TotalPremium'], errors='coerce')
    data['Province'] = data['Province'].astype('category')
    data['CoverCategory'] = data['CoverCategory'].astype('category')

    plt.figure(figsize=(15, 10))
    sns.boxplot(data=data, x='Province', y='TotalPremium', hue='CoverCategory', palette='viridis')
    plt.title('Distribution of Total Premium by Province and Cover Category')
    plt.xlabel('Province')
    plt.ylabel('Total Premium')
    plt.xticks(rotation=45)  
    plt.legend(title='CoverCategory', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_vehicles_by_province_and_make(data):
    
    data['make'] = data['make'].astype('category')

    plt.figure(figsize=(15, 10))
    sns.countplot(data=data, x='Province', hue='make', palette='viridis')
    plt.title('Count of Vehicles by Province and Make')
    plt.xlabel('Province')
    plt.ylabel('Count')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.legend(title='Vehicle Make', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_total_premium_trends(data):
    
    # Convert 'TransactionMonth' to datetime
    data['TransactionMonth'] = pd.to_datetime(data['TransactionMonth'], errors='coerce')

    # Create the line plot
    plt.figure(figsize=(15, 10))
    sns.lineplot(data=data, x='TransactionMonth', y='TotalPremium', hue='Province', palette='viridis')
    plt.title('Total Premium Trends Over Time by Province')
    plt.xlabel('Month')
    plt.ylabel('Total Premium')
    plt.tight_layout()
    plt.show()

def plot_numerical_boxplots(data, numerical_columns):

    num_plots = len(numerical_columns)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols

    plt.figure(figsize=(15, 5 * num_rows))

    for i, column in enumerate(numerical_columns, 1):
        plt.subplot(num_rows, num_cols, i)
        sns.boxplot(data=data, y=column)
        plt.title(f'Box Plot of {column}')

    plt.tight_layout()
    plt.show()

def plot_violin_distribution(data, x_column, y_column, hue_column):
    
    plt.figure(figsize=(15, 25))
    sns.violinplot(data=data, x=x_column, y=y_column, hue=hue_column, palette='viridis', legend=False)
    plt.title('Distribution of Insurance Premiums by Vehicle Make')
    plt.xticks(rotation=90)
    plt.xlabel('Vehicle Make')
    plt.ylabel('Total Premium')
    plt.tight_layout()
    plt.show()

def plot_scatter_with_regression(data, x_column, y_column, hue_column, palette='viridis'):
    
    plt.figure(figsize=(15, 8))
    sns.scatterplot(data=data, x=x_column, y=y_column, hue=hue_column, palette=palette, alpha=0.7)
    sns.regplot(data=data, x=x_column, y=y_column, scatter=False, color='black')
    plt.title(f'{y_column} vs. {x_column} by {hue_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend(title=hue_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_boxplot_by_category(data, x_column, y_column, hue_column, palette='viridis'):
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=data, x=x_column, y=y_column, hue=hue_column, palette=palette, dodge=False)
    plt.title(f'Box Plot of {y_column} by {x_column}')
    plt.xticks(rotation=45)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.tight_layout()
    plt.show()

