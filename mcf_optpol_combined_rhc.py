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
PATH = "F:/fmascolo/rhc/data/"

# This is the the only df with the cleaning applied for the pseudo dates
alldata_df = pd.read_csv(PATH + "rhc.csv")


# The outcome is recoded for interpretability 
alldata_df['dth30'].value_counts()
alldata_df['dth30'] = np.where(alldata_df['dth30'] == 1, 0, 1)

alldata_df['dth30'].value_counts()

# Round surviv. probability
# alldata_df.surv2md1 = alldata_df.surv2md1.round(2)

# -----------------------------------------------------------------------------
# Split the data

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
APPLIC_PATH = "F:/fmascolo/rhc/data/"
DATPATH = APPLIC_PATH + '/data'
# ALLDATA = 'data_y_d_x_4000.csv'
# Training data must contain outcome, treatment and features.
OUTPUT_PATH = "F:/fmascolo/rhc/"


# ------------------ Reset some parameters  -------------------
VAR_D_NAME = 'swang1'   # Name of treatment variable
VAR_Y_NAME = 'dth30'         # Name of outcome variable


VAR_POLSCORE_NAME = ('dth30_LC0_un_lc_pot', 'dth30_LC1_un_lc_pot',)


# ordered variables
var_x_name_ord = ['adld3pc', 'age', 'alb1', 'amihx', 'aps1', 'bili1',
                    'card', 'cardiohx', 'cat2_miss', 'chfhx',
                    'chrpulhx', 'crea1', 'das2d3pc', 'dementhx', 'edu',
                    'gastr', 'gibledhx', 'hema', 'hema1', 'hrt1', 'immunhx',
                    'income', 'liverhx', 'malighx', 'meanbp1', 'meta', 'neuro',
                    'ortho', 'paco21', 'pafi1', 'ph1', 'pot1', 'psychhx',
                    'renal', 'renalhx', 'resp', 'resp1', 'scoma1', 'seps', 'sex',
                    'sod1', 'surv2md1', 'temp1', 'transhx', 'trauma', 'urin1',
                    'urin1_miss', 'wblc1', 'wtkilo1' , 'dnr1']


# Unordered variables
var_x_name_unord = ['ca', 'cat1', 'cat2', 'ninsclas', 'race']


# Variables always included

important_x_name_ordered = ['adld3pc', 'age', 'aps1',
                            'meanbp1', 'surv2md1', 'scoma1']

important_x_name_unordered = ['cat1']


# var_z_name_list = important_x_name_ordered[:5]
# var_z_name_ord = important_x_name_ordered[5:]
# var_z_name_unord = important_x_name_unordered



# Optimal policy vars
VAR_X_NAME_ORD_OP = ['adld3pc', 'age', 'aps1', 'scoma1',
                     'meanbp1', 'surv2md1', 'dnr1']
VAR_X_NAME_UNORD_OP = ['cat1' ]



# ------------------------- Modified Causal Forest ----------------------------

# Initialize the ModifiedCausalForest
mymcf = ModifiedCausalForest(var_d_name=VAR_D_NAME,
                             var_y_name=VAR_Y_NAME,
                             #var_z_name_list=var_z_name_list,
                             var_x_name_ord=var_x_name_ord,
                             var_x_name_unord=var_x_name_unord,
                             #var_z_name_ord=var_z_name_ord,
                             #var_z_name_unord=var_z_name_unord,
                             p_atet=True, 
                             p_ci_level=0.95, p_iate_se=True,
                             gen_iate_eff=True,
                             gen_outpath=OUTPUT_PATH,
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


# # # Save 'results' as CSV
results['iate_data_df'].to_csv(DATPATH +'results_train.csv', index=False)

# Save 'results_oos' as CSV
results_oos['iate_data_df'].to_csv(DATPATH + 'results_oos.csv', index=False)

# Save 'data_train_pt' as CSV
data_train_pt.to_csv(DATPATH + 'data_train_pt.csv', index=False)

# Save 'oos_df' as CSV
oos_df.to_csv(DATPATH + 'oos_df.csv', index=False)

# Load the data
results_train = pd.read_csv(DATPATH + 'results_train.csv')
results_oos = pd.read_csv(DATPATH + 'results_oos.csv')

data_train_pt = pd.read_csv(DATPATH + 'data_train_pt.csv')
oos_df = pd.read_csv(DATPATH + 'oos_df.csv')

# ------------------------- Policy tree ---------------------------------

# --- Parameters --

GEN_METHOD = 'policy tree'
pt_depth_tree1_values = [2, 3] 
pt_depth_tree2_values = [0, 1]
# other_max_shares_values = [[1, 1], [1, 0.38]]
# cost_multiplier = [[1, 1], [1, 1.03]]


for pt_depth_tree_1 in pt_depth_tree1_values:
    for pt_depth_tree_2 in pt_depth_tree2_values:
        # for other_max_shares in other_max_shares_values:
        #     if other_max_shares==other_max_shares_values[0]:
        #         cost_mult = cost_multiplier[0]
        #     else:
        #         cost_mult = cost_multiplier[1]                
            myoptp = OptimalPolicy(var_d_name=VAR_D_NAME,
                       var_x_name_ord=VAR_X_NAME_ORD_OP,
                       var_x_name_unord = VAR_X_NAME_UNORD_OP,
                       var_polscore_name=VAR_POLSCORE_NAME,
                       pt_depth_tree_1=pt_depth_tree_1,
                       pt_depth_tree_2=pt_depth_tree_2,
                       gen_outpath=OUTPUT_PATH,
                       gen_method=GEN_METHOD  #,
                       # other_max_shares=other_max_shares,
                       # other_costs_of_treat_mult = cost_mult
                       )
            
            # Train the optimal tree
            alloc_train_df = myoptp.solve(data_train_pt,
                                          data_title='Training PT data')
            
            results_eva_train = myoptp.evaluate(alloc_train_df, data_train_pt,
                                                data_title='Training PT data')
            # Evaluation is out of sample
            alloc_eva_df = myoptp.allocate(oos_df, data_title='')
            
            results_eva_oos = myoptp.evaluate(alloc_eva_df, oos_df,
                                                data_title='Evaluate PT data')
            
            myoptp.print_time_strings_all_steps()


my_report = McfOptPolReport(mcf=mymcf, optpol=myoptp,
                            outputfile='Report_mcf_optpolicy')
my_report.report()


print('End of computations.\n\nThanks for using ModifiedCausalForest and'
      ' OptimalPolicy. \n\nYours sincerely\nMCF \U0001F600')


#Only one policy tree
GEN_METHOD = 'policy tree'
pt_depth_tree1_values = [2,] 
pt_depth_tree2_values= [1,]
other_max_shares_values = [[1, 1]]#, [1,1]]
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
            
            results_eva_oos = myoptp.evaluate(alloc_eva_df, oos_df,
                                                data_title='Evaluate PT data')
            
            myoptp.print_time_strings_all_steps()


my_report = McfOptPolReport(mcf=mymcf, optpol=myoptp,
                            outputfile='Report_mcf_optpolicy')
my_report.report()
