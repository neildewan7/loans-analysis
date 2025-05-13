# Loan Applications Analysis 💸

This project explores and analyzes a real-world dataset of approved loan applications from LendingClub. It investigates how borrower characteristics influence loan terms, interest rates, and risk assessment by lenders.

## Project Overview

- **Data Cleaning:** 
  - Converted string columns like issue dates and terms into usable formats.
  - Standardized employment titles and computed loan term end dates.
  
- **Exploratory Analysis:**
  - Investigated how debt-to-income (DTI) ratios relate to interest rates offered.
  - Analyzed borrower income distributions across different loan terms.
  - Compared interest rates and incomes for different loan purposes.

- **Income Adjustment:** 
  - Estimated borrowers' disposable incomes after monthly debt payments to better assess financial health.

- **Statistical Insights:** 
  - Detected instances of **Simpson's Paradox** when examining annual income and interest rates across different loan purposes.
  
- **Visualization:** 
  - Used Plotly to create interactive scatterplots, histograms, and density plots to reveal important trends.

## Demo

👉 [**View Project Demo (nbviewer)**](https://nbviewer.org/github/neildewan7/loans-analysis/blob/main/cleaned_project.ipynb)

## Key Questions Explored

- How does a borrower's debt-to-income ratio (DTI) influence their interest rate?
- Do borrowers with longer loan terms tend to have different financial profiles?
- How does disposable income affect loan terms and interest rates?
- Can Simpson's Paradox appear when comparing interest rates across different borrower groups?

## Technologies Used

- **Python**
- **Pandas** (data manipulation)
- **Plotly** (interactive visualizations)
- **NumPy** (numerical calculations)

## Project Structure

```plaintext
cleaned_project.ipynb    # Final cleaned notebook with full analysis
project.py               # Helper functions for data manipulation and feature engineering
data/                    # Folder containing the loans dataset
images/                  # (Optional) Folder for generated visualizations
README.md                # Project overview and instructions
