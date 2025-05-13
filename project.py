# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pd.options.plotting.backend = 'plotly'

from IPython.display import display

# DSC 80 preferred styles
pio.templates["dsc80"] = go.layout.Template(
    layout=dict(
        margin=dict(l=30, r=30, t=30, b=30),
        autosize=True,
        width=600,
        height=400,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        title=dict(x=0.5, xanchor="center"),
    )
)
pio.templates.default = "simple_white+dsc80"
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def clean_loans(loans):

    #change to datetime
    loans['issue_d'] = pd.to_datetime(loans['issue_d'])

    #change to int
    loans['term'] = loans['term'].str.replace('months', '').astype(int)

    #change rn
    loans['emp_title'] = loans['emp_title'].str.strip().str.lower().transform(lambda job: 'registered_nurse' if job == 'rn' else job)

    #add col
    loans['term_end'] = loans.apply(lambda row: row['issue_d'] + pd.DateOffset(months=row['term']), axis=1)

    return loans

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def correlations(df, pairs):

    dic = {}
    
    for col1, col2 in pairs:

        correlation = df[col1].corr(df[col2])

        dic[f"r_{col1}_{col2}"] = correlation

    return pd.Series(dic)




# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def create_boxplot(loans):

    bins = [580, 670, 740, 800, 850]
    labels = ['[580, 670)', '[670, 740)', '[740, 800)', '[800, 850)']

    
    loans['credit_score_bin'] = pd.cut(loans['fico_range_low'], bins=bins, right=False, labels=labels)


    loans['term'] = loans['term'].astype(str)


    fig = px.box(loans,
                 x='credit_score_bin',
                 y='int_rate',
                 color='term',
                 color_discrete_map={'36': 'purple', '60': 'gold'},
                 category_orders={'credit_score_bin': labels},
                 labels={
                     'int_rate': 'Interest Rate (%)',
                     'term': 'Loan Length (Months)'
                 },
                 title='Interest Rate vs. Credit Score')

    fig.update_layout(
        xaxis_title='Credit Score Range',
        yaxis_title='Interest Rate (%)',
        legend_title='Loan Length (Months)'
    )
    
    return fig



# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def ps_test(loans, N):

    #create a col for that contains bool indicating if entry has a ps
    loans['has_ps'] = loans['desc'].notna()

    #means 
    mean_with_ps = loans[loans['has_ps']]['int_rate'].mean()
    mean_no_ps = loans[loans['has_ps'] == False]['int_rate'].mean()

    #test statistic : difference in group means
    shuffled_loans = loans.copy()
    test_stat = mean_with_ps - mean_no_ps
    permutations = []
    
    for i in range(N):

        #shuffle 'has_ps' col'
        shuffled_loans['has_ps'] = np.random.permutation(shuffled_loans['has_ps'])

        #means pof shuffled
        perm_ps_mean = shuffled_loans[shuffled_loans['has_ps']]['int_rate'].mean()
        perm_no_ps_mean = shuffled_loans[shuffled_loans['has_ps'] == False]['int_rate'].mean()

        #permutated test statistic
        permutation_gmean = perm_ps_mean - perm_no_ps_mean
        permutations.append(permutation_gmean)

    #calculate p value
    p_val = (permutations >= test_stat).mean()

    return p_val

def missingness_mechanism():
    return 2
    
def argument_for_nmar():
    string = '''
    An argument is that applicants might not submit personal 
    statements because of unobserved factors,
    which can also influence the interest rates they receive. 
    This creates scenarios where missingness can depend on unobserved 
    variables, making it have the possibility of being NMAR.
    
    '''
    return string


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def tax_owed(income, brackets):
    total = 0.0
    for i in range(len(brackets)):
        rate, lower = brackets[i]
        # Get upper limit (next bracket's lower or infinity)
        upper = brackets[i+1][1] if i < len(brackets)-1 else float('inf')
        taxable = max(0, min(income, upper) - lower)
        total += taxable * rate
    return total

# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_state_taxes(state_taxes_raw): 

    taxes = state_taxes_raw.copy()

    def drop_empty_rows(df: pd.DataFrame):
        return df[df.isna().all(axis = 1) == False]

    def state_cleaned(df: pd.DataFrame):
        df['State'] = df['State'].astype(str).apply(lambda x: np.nan if '(' in x or 'nan' in x else x).fillna(method = 'ffill')
        return df

    def rate_cleaned(df: pd.DataFrame):

        df['Rate'] = df['Rate'].replace('none', np.nan).str.replace('%', '').fillna(0.0)

        df['Rate'] = df['Rate'].astype(float) / 100

        df['Rate'] = df['Rate'].round(2)

        return df

    def lower_limit_cleaned(df: pd.DataFrame):

        df['Lower Limit'] = df['Lower Limit'].apply(
            lambda x: int(x.replace('$', '').replace(',','')) if pd.notna(x) else 0)
        return df

    return taxes.pipe(drop_empty_rows
            ).pipe(state_cleaned
            ).pipe(rate_cleaned
            ).pipe(lower_limit_cleaned)




# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def state_brackets(state_taxes):
    return (state_taxes.groupby('State')[['Rate', 'Lower Limit']]
            .apply(lambda df: list(zip(df['Rate'], df['Lower Limit'])))
            .reset_index()
            .rename(columns={0: 'bracket_list'})
            .set_index('State'))
    
def combine_loans_and_state_taxes(loans, state_taxes):
    import json

    state_mapping_path = Path('data') / 'state_mapping.json'
    with open(state_mapping_path, 'r') as f:
        state_mapping = json.load(f)

    state_taxes['State'] = state_taxes['State'].map(state_mapping)

    state_tax_brackets = state_brackets(state_taxes)

    loans_taxes = loans.rename(columns={'addr_state': 'State'}).merge(
        state_tax_brackets, on='State', how='left'
    )

    loans_taxes['bracket_list'] = loans_taxes['bracket_list'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    return loans_taxes

# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def find_disposable_income(loans_with_state_taxes):

    FEDERAL_BRACKETS = [
        (0.1, 0), 
        (0.12, 11000), 
        (0.22, 44725), 
        (0.24, 95375), 
        (0.32, 182100),
        (0.35, 231251),
        (0.37, 578125)
    ]

    df = loans_with_state_taxes.copy()

    df['federal_tax_owed'] = df['annual_inc'].apply(lambda inc: tax_owed(inc, FEDERAL_BRACKETS))

    df['state_tax_owed'] = df.apply(lambda row: tax_owed(row['annual_inc'], row['bracket_list']), axis=1)

    df['disposable_income'] = df['annual_inc'] - df['federal_tax_owed'] - df['state_tax_owed']

    return df


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def aggregate_and_combine(loans, keywords, quantitative_column, categorical_column):
    df = pd.DataFrame()

    for keyword in keywords:

        keyword_filter = loans['emp_title'].str.contains(keyword, na=False)
        
        means = loans[keyword_filter].groupby(categorical_column)[quantitative_column].mean()

        overall_mean = loans[keyword_filter][quantitative_column].mean()

        means['Overall'] = overall_mean
        
        means.name = f"{keyword}_mean_{quantitative_column}"
    
        df = pd.concat([df, means], axis=1)

    return df


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def exists_paradox(loans, keywords, quantitative_column, categorical_column):

    aggregated = aggregate_and_combine(loans, keywords, quantitative_column, categorical_column)

    paradox_found = ((aggregated.iloc[:-1, 0] > aggregated.iloc[:-1, 1]).all() and
            (aggregated.iloc[-1, 0] < aggregated.iloc[-1, 1]))
    

    return bool(paradox_found)


def paradox_example(loans):

    keywords = [['manager', 'assistant'], ['doctor', 'teacher'], ['developer', 'analyst']]
    quant_cols = ['loan_amnt', 'int_rate', 'annual_inc']
    categ_cols = ['home_ownership', 'verification_status', 'term']

    for keyword_pair in keywords:
        for quant_col in quant_cols:
            for categ_col in categ_cols:
                if exists_paradox(loans, keyword_pair, quant_col, categ_col):
                    return {
                        'loans': loans,
                        'keywords': keyword_pair,
                        'quantitative_column': quant_col,
                        'categorical_column': categ_col
                    }
    return None
