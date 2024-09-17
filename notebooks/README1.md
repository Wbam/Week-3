# Insurance Claim Data Analysis - Task 3: A/B Hypothesis Testing

## Overview

This project involves analyzing historical insurance claim data to optimize marketing strategies and identify potential low-risk targets for reduced premiums. Task 3 focuses on performing A/B hypothesis testing to evaluate various hypotheses related to risk and profit margins.

## Objectives

1. **Test Risk Differences Across Provinces**: Evaluate whether there are significant differences in risk between different provinces.
2. **Test Risk Differences Between Zip Codes**: Determine if there are significant differences in risk between specific zip codes.
3. **Test Profit Margin Differences Between Zip Codes**: Assess whether profit margins differ significantly between zip codes.
4. **Test Risk Differences Between Genders**: Examine if there is a significant difference in risk between men and women.

## Data

The data used in this analysis includes historical insurance claim data from February 2014 to August 2015. Key columns of interest are:

- `TotalClaims`: The total claims made.
- `PostalCode`: Zip code of the client.
- `Province`: Province of the client.
- `Gender`: Gender of the client.

## Methodology

1. **Data Loading and Preprocessing**

   - Load the dataset.
   - Clean the data by handling missing values.

2. **Data Segmentation**

   - For each hypothesis, segment the data into control and test groups based on the feature being tested.

3. **Statistical Testing**

   - Use T-tests for numerical data (risk and profit margins).
   - Use Chi-squared tests for categorical data (gender risk differences).

4. **Hypothesis Testing**
   - Test the following null hypotheses:
     1. No risk differences across provinces.
     2. No risk differences between zip codes.
     3. No significant profit margin differences between zip codes.
     4. No significant risk differences between genders.

## Recommendations

- **Data Review**: Verify the completeness of the data for zip code analysis.
- **Further Analysis**: Consider exploring additional features or different segmentation strategies if needed.
- **Business Strategy**: Use findings to influence marketing and pricing strategies.

## Running the Analysis

1. Ensure that the dataset is located in the correct path.
2. Install required Python packages:
   ```bash
   pip install pandas scipy
   ```
