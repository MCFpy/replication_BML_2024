# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:55:39 2024

@author: p1ufederica.masc
"""

"""Created on Wed Apr  1 15:58:30 2020. -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.5.0

This is an example to show how to combine the ModifiedCausalForest class and
the OptimalPolicy class for joint estimation. Please note that there could be
many alternative ways to split sample, including full cross-validation, that
may be superior to the simple split used here.

"""

import pandas as pd
import numpy as np
import statistics
from mcf.mcf_functions import ModifiedCausalForest
from mcf.optpolicy_functions import OptimalPolicy
from mcf.reporting import McfOptPolReport
from sklearn.model_selection import train_test_split


#  In this example we combine mcf estimation and an optimal policy tree in a
#  simple split sample approach.
#  Split alldata_df into 3 random samples of equal number of rows:
#  train_mcf_df: Train mcf.
#  pred_mcf_train_pt_df: Predict IATEs and train policy score.
#  evaluate_pt_df: Evaluate policy score.

# Integrate code
PATH = "F:/fmascolo/ce/data/"

# This is the the only df with the cleaning applied for the pseudo dates
df = pd.read_stata(PATH + "kz_demandelasts_aer08.dta")

# check variables from original specification
df.applied.value_counts(dropna=False)

df.offer4.describe()
# create histogram of price offers
df.offer4.hist()


# create the treatment based on stardard risks
df['offer_risk'] = np.nan
df.loc[df.offer4 <= 7.75, 'offer_risk'] = 1 
df.loc[(df.offer4 > 7.75) & (df.offer4 <= 9.75), 'offer_risk'] = 2
df.loc[df.offer4 > 9.75, 'offer_risk'] = 3


# Data cleaning
df.branchuse = pd.factorize(df['branchuse'])[0] + 1

df["risk_num"] = 1
df.loc[df.risk == "MEDIUM", "risk_num"] = 2
df.loc[df.risk == "HIGH", "risk_num"] = 3

df['grossincome'].isna().sum()

df = df[~df.grossincome.isna()]

# # create quintiles of price offers

quantiles = df['grossincome'].quantile([0, 0.25, 0.5, 0.75, 1])
print(quantiles)

labels = quantiles.iloc[1:].tolist()

# Assign values to quantiles based on ue_rate_region
df['qnt_income'] = pd.cut(df['grossincome'],
                              bins=quantiles, labels=False,
                              include_lowest=True)

df['qnt_income'].value_counts(dropna=False)


# create a new outcome
df['adj_loan'] = df.final4 * df.loansize

df.adj_loan.value_counts(dropna=False)


df1 = df[['offer_risk', 'branchuse', 'wave', 'applied', 'risk_num',
          'age', 'female', 'married', 'edhi', 'rural', 'dependants',
          'qnt_income', 'adj_loan']]

df1.isna().sum()

df1 = df1.dropna()

df1.age = df.age.round(0)

df1.dtypes

# recode the treatment 

df1.offer_risk.value_counts()

df1.loc[df1.offer_risk == 3, 'offer_risk'] = 0
#df1.loc[df1.offer_risk == 3, 'offer_risk'] = 2


# -----------------------------------------------------------------------------
# Split the data

alldata_df = df1.copy()


def split_data(df, train_size=0.4, validation_size=0.2, 
               test_size=0.4, random_state=None):

    # Calculate the size of each portion
    total_size = train_size + validation_size + test_size
    train_size = train_size / total_size
    validation_size = validation_size / total_size
    
    # Split the DataFrame into train, validation, and test sets
    train_df, remaining_df = train_test_split(df, train_size=train_size,
                                              random_state=random_state)
    validation_df, test_df = train_test_split(remaining_df,
                                              train_size=validation_size, 
                                              random_state=random_state)
    
    return train_df, validation_df, test_df


train_mcf_df, evaluate_pt_df, pred_mcf_train_pt_df = split_data(alldata_df,
                                                                random_state=42)

# Reset index
train_mcf_df.reset_index(drop=True, inplace=True)
pred_mcf_train_pt_df.reset_index(drop=True, inplace=True)
evaluate_pt_df.reset_index(drop=True, inplace=True)

# -----------------------------------------------------------------------------


# Step 1: Define data to be used in this example
APPLIC_PATH = "F:/fmascolo/ce/data/"
DATPATH = APPLIC_PATH + '/data'
# Training data must contain outcome, treatment and features.
OUTPUT_PATH = "F:/fmascolo/ce/"


# ------------------ Reset some parameters  -------------------
VAR_D_NAME = 'offer_risk'  
VAR_Y_NAME = 'adj_loan'         


VAR_POLSCORE_NAME = ('adj_loan_LC0_un_lc_pot',
                     'adj_loan_LC1_un_lc_pot',
                     'adj_loan_LC2_un_lc_pot')


# ordered variables  
var_x_name_ord = ['age', 'dependants', 'qnt_income',
                  'wave', 'risk_num', 'female', 'married',
                  'edhi', 'rural', ]


# Unordered variables
var_x_name_unord = []

# Variables always included

important_x_name_ordered = []

important_x_name_unordered = []


# Gates variables
var_z_name_list = ['age','qnt_income','female', 'rural', 'edhi']
var_z_name_ord = []
var_z_name_unord = []

# why female yes and race no? medical conditiona for women are different


# Optimal policy vars
VAR_X_NAME_ORD_OP = ['age','qnt_income', 'female', 'rural', 'edhi']
VAR_X_NAME_UNORD_OP = []




# ------------------------- Modified Causal Forest ----------------------------

# Initialize the ModifiedCausalForest
mymcf = ModifiedCausalForest(var_d_name=VAR_D_NAME,
                             var_y_name=VAR_Y_NAME,
                             var_x_name_ord=var_x_name_ord,
                             #var_x_name_unord=var_x_name_unord,
                             var_z_name_list=var_z_name_list,
                             var_z_name_ord=var_z_name_ord,
                            # var_z_name_unord=var_z_name_unord,
                             p_atet=True, 
                             p_ci_level=0.95, p_iate_se=True,
                             gen_iate_eff=True,
                             gen_outpath=OUTPUT_PATH,
                             var_cluster_name="branchuse",
                             _int_show_plots=False)


mymcf.train(train_mcf_df)


results = mymcf.predict(pred_mcf_train_pt_df)
mymcf.analyse(results)

# create the oos results
results_oos = mymcf.predict(evaluate_pt_df)

# Train of the policy tree is the predict of the mcf
data_train_pt = results['iate_data_df']
# out of sample?
oos_df = results_oos['iate_data_df']



# # Save 'results' as CSV
# results['iate_data_df'].to_csv(DATPATH +'results_train.csv', index=False)

results_train = pd.read_csv(DATPATH +'results_train.csv')


# # Save 'results_oos' as CSV
# results_oos['iate_data_df'].to_csv(DATPATH + 'results_oos.csv', index=False)

results_oos = pd.read_csv(DATPATH + 'results_oos.csv')

# # Save 'data_train_pt' as CSV
# data_train_pt.to_csv(DATPATH + 'data_train_pt.csv', index=False)

data_train_pt = pd.read_csv(DATPATH + 'data_train_pt.csv')


# # Save 'oos_df' as CSV
# oos_df.to_csv(DATPATH + 'oos_df.csv', index=False)

oos_df = pd.read_csv(DATPATH + 'oos_df.csv')


# ------------------------- Policy tree ---------------------------------

# --- Parameters --

GEN_METHOD = 'policy tree'
pt_depth_tree1_values = [2, 3, 4] 
pt_depth_tree2_values= [0, 1]
other_max_shares_values = [[1,1,], [1, 0.38]]
#other_costs_of_treat_mult = [1, 1.03]



for pt_depth_tree_1 in pt_depth_tree1_values:
    for pt_depth_tree_2 in pt_depth_tree2_values:
        for other_max_shares in other_max_shares_values:
            myoptp = OptimalPolicy(var_d_name=VAR_D_NAME,
                       var_x_name_ord=VAR_X_NAME_ORD_OP,
                       var_x_name_unord = VAR_X_NAME_UNORD_OP,
                       var_polscore_name=VAR_POLSCORE_NAME,
                       pt_depth_tree_1=pt_depth_tree_1,
                       pt_depth_tree_2=pt_depth_tree_2,
                       gen_outpath=OUTPUT_PATH,
                       gen_method=GEN_METHOD,
                       other_max_shares = other_max_shares,
                       )  #other_costs_of_treat_mult
            # Train the optimal tree
            alloc_train_df = myoptp.solve(data_train_pt, data_title='Training PT data')
            
            results_eva_train = myoptp.evaluate(alloc_train_df, data_train_pt,
                                                data_title='Training PT data')
            # Evaluation is out of sample
            alloc_eva_df = myoptp.allocate(oos_df, data_title='')
            
            results_eva_train = myoptp.evaluate(alloc_eva_df, oos_df,
                                                data_title='Evaluate PT data')
            
            myoptp.print_time_strings_all_steps()


my_report = McfOptPolReport(mcf=mymcf, optpol=myoptp,
                            outputfile='Report_mcf_optpolicy')
my_report.report()


print('End of computations.\n\nThanks for using ModifiedCausalForest and'
      ' OptimalPolicy. \n\nYours sincerely\nMCF \U0001F600')



GEN_METHOD = 'policy tree'
pt_depth_tree1_values = [4] 
pt_depth_tree2_values= [2]
other_max_shares_values = [[1, 1, 1]]#, [1, 0.38]]
#other_costs_of_treat_mult = [1, 1, 1.03]
# 0.40, 0.29, 0.31]

for pt_depth_tree_1 in pt_depth_tree1_values:
    for pt_depth_tree_2 in pt_depth_tree2_values:
        for other_max_shares in other_max_shares_values:
            myoptp = OptimalPolicy(var_d_name=VAR_D_NAME,
                       var_x_name_ord=VAR_X_NAME_ORD_OP,
                       var_x_name_unord = VAR_X_NAME_UNORD_OP,
                       var_polscore_name=VAR_POLSCORE_NAME,
                       pt_depth_tree_1=pt_depth_tree_1,
                       pt_depth_tree_2=pt_depth_tree_2,
                       gen_outpath=OUTPUT_PATH,
                       gen_method=GEN_METHOD,
                       other_max_shares = other_max_shares,
                       #other_costs_of_treat_mult = [1, 1.27, 1.3]
                       )  # 

            # Train the optimal tree
            alloc_train_df = myoptp.solve(data_train_pt, data_title='Training PT data')
            
            results_eva_train = myoptp.evaluate(alloc_train_df, data_train_pt,
                                                data_title='Training PT data')
            # Evaluation is out of sample
            alloc_eva_df = myoptp.allocate(oos_df, data_title='')
            
            results_eva_train = myoptp.evaluate(alloc_eva_df, oos_df,
                                                data_title='Evaluate PT data')
            
            myoptp.print_time_strings_all_steps()


my_report = McfOptPolReport(mcf=mymcf, optpol=myoptp,
                            outputfile='Report_mcf_optpolicy')
my_report.report()


print('End of computations.\n\nThanks for using ModifiedCausalForest and'
      ' OptimalPolicy. \n\nYours sincerely\nMCF \U0001F600')

