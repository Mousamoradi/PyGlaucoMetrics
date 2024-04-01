Instruction:
1- Open Anaconda PowerShell Prompt:
>> cd "set path to folder"
2- conda activate env_pyVF
#3-Check if R-package and rpy2 installed already
# python  test_rpy2.py# Instruction:

1- Open Anaconda PowerShell Prompt:

>> cd "set path to folder"

2- conda activate env_pyVF

#3-Check if R-package and rpy2 installed already

#python  test_rpy2.py

4- Open Jupyter notebook

>> jupyter notebook




# General overview:
# 1- Import raw VF data
df_VFs = pd.read_csv('VF_Data.csv')
# 2- Get td, tdp, pd, and pdp from PyVisualField Package. 
df_td, df_tdp, df_pdp = visualFields.getallvalues(df_VFs) 
# 3- Obtain required columns from each dataframe
raw_data_pdp = df_pdp.loc[:, 'l1':'l54']
raw_data_td = df_td.loc[:, 'l1':'l54']
raw_data_tdp = df_tdp.loc[:, 'l1':'l54']
# 4- CAll each function and save resulted diagnosis 
df_diag_HAP2 = Fn_HAP2_part2(raw_data_pdp) # it needs pdp values. will compute if necessary
df_diag_UKG = Fn_UKGTS(raw_data_td) #it needs tdp values, will compute if necessary
df_diag_logts = Fn_LoGTS(raw_data_tdp) # it need TD values, will compute if necessary



# References:
1- "PyGlaucoMetrics: An Open-Source Multi-Criteria Glaucoma Defect Evaluation", ARVO 2024, Accepted Abstract

2- "PyVisualFields: A Python Package for Visual Field Analysis", https://tvst.arvojournals.org/article.aspx?articleid=2785341)https://tvst.arvojournals.org/article.aspx?articleid=2785341





# General overview:
# 1- Import raw VF data
df_VFs = pd.read_csv('VF_Data.csv')
# 2- Get td, tdp, pd, and pdp from PyVisualField Package. 
df_td, df_tdp, df_pdp = visualFields.getallvalues(df_VFs) 
# 3- Obtain required columns from each dataframe
raw_data_pdp = df_pdp.loc[:, 'l1':'l54']
raw_data_td = df_td.loc[:, 'l1':'l54']
raw_data_tdp = df_tdp.loc[:, 'l1':'l54']
# 4- CAll each function and save resulted diagnosis 
df_diag_HAP2 = Fn_HAP2_part2(raw_data_pdp) # it needs pdp values. will compute if necessary
df_diag_UKG = Fn_UKGTS(raw_data_td) #it needs tdp values, will compute if necessary
df_diag_logts = Fn_LoGTS(raw_data_tdp) # it need TD values, will compute if necessary
