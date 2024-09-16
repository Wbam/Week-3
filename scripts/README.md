# Insurance Data Exploratory Data Analysis (EDA)

## Project Overview

This project performs Exploratory Data Analysis (EDA) on an insurance dataset. The objective is to gain insights into the data by understanding its structure, quality, and relationships between variables. This analysis includes detecting outliers, assessing data quality, and generating visualizations to summarize findings.

---

## Analysis Steps

### 1. Data Summarization

- **Descriptive Statistics**:
  - We calculated descriptive statistics such as mean, standard deviation, and quartiles for numerical variables like `TotalPremium`, `TotalClaim`, etc. This helps in understanding the central tendency and spread of the data.
- **Data Structure**:
  - The data types of each column were reviewed to ensure that categorical, numerical, and date variables were properly formatted. Categorical columns like `CoverCategory` and numerical columns like `TotalPremium` were inspected for correctness.

### 2. Data Quality Assessment

- **Missing Values**:
  - We checked for missing data across the dataset. Missing value imputation was considered for minor gaps, and missing rows were handled accordingly based on their impact on analysis.

### 3. Univariate Analysis

- **Distribution of Variables**:
  - We plotted histograms for numerical columns like `TotalPremium` and `TotalClaim`, and bar charts for categorical columns such as `CoverCategory`. This visualizes the distribution and frequency of different values, offering insight into data patterns.

### 4. Bivariate/Multivariate Analysis

- **Correlations and Associations**:

  - Relationships between numerical variables (e.g., `TotalPremium` and `TotalClaim`) were explored using scatter plots and correlation matrices.
  - Key correlations were examined by geographic areas like `ZipCode`, enabling a comparison of premiums and claims across regions.

- **Geographical Trends**:
  - We compared changes in variables such as `Insurance Cover Type`, `Premium`, and `Auto Make` over different geographical areas, providing insight into trends based on location.

### 5. Outlier Detection

- **Box Plots**:
  - We used box plots to detect and analyze outliers in numerical columns such as `TotalPremium` and `TotalClaim`. Outliers were further examined for their potential impact on the overall analysis.

### 6. Data Visualization

- **Creative Visualizations**:
  - Several insightful plots were generated to highlight key patterns and trends in the data. Notably:
    - **Box Plot**: A box plot comparing `TotalPremium` across different `CoverCategories`.
    - **Scatter Plot**: A scatter plot showcasing the relationship between `TotalPremium` and `TotalClaim` over `ZipCode`.
    - **Correlation Matrix**: A heatmap showing the correlation between all numerical features in the dataset.
