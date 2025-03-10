# Description: PyGlaucoMetrics is designed to classify visual field data to glaucomatous or non-glaucomatous classes. It can accept Humphrey Field Analysis 24-2 and 10-2 test patterns. PyGlaucometrics, can specifically be useful for brlow data types:
# (A) data_vfpwgRetest24d2: Short-term retest static automated perimetry data, collected from 30 glaucoma patients at the Queen Elizabeth Health Sciences Centre in Halifax, Nova Scotia. This dataset includes 12 visual field tests conducted over 12 weekly sessions.
# (B) data_vfpwgSunyiu24d2: 24-2 static automated perimetry data from a patient with glaucoma. This dataset consists of real patient data, with age modified for anonymity.
# (C) data_vfctrSunyiu24d2: A dataset of healthy eyes for 24-2 static automated perimetry, used to generate normative values. This dataset (sunyiu_24d2 and related sets) is provided courtesy of William H. Swanson and Mitch W. Dul.
# (D) data_vfctrSunyiu10d2: A dataset of healthy eyes for 10-2 static automated perimetry, also contributed by William H. Swanson.

# Instruction:

***This instruction assumes the required library correctly installed. If not, please install exact version of libraries/packages as stated in the Requirements.

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
# 4- Call each function and save resulted diagnosis 
df_diag_HAP2 = Fn_HAP2_part2(raw_data_pdp) # it needs pdp values. will compute if necessary
df_diag_UKG = Fn_UKGTS(raw_data_td) #it needs tdp values, will compute if necessary
df_diag_logts = Fn_LoGTS(raw_data_tdp) # it need TD values, will compute if necessary



# References:
1- Moradi, M., Eslami, M., Hashemabad, S.K., Friedman, D.S., Boland, M.V., Wang, M., Elze, T. and Zebardast, N., 2024. PyGlaucoMetrics: An Open-Source Multi-Criteria Glaucoma Defect Evaluation. Investigative Ophthalmology & Visual Science, 65(7), pp.OD38-OD38.

2- "PyVisualFields: A Python Package for Visual Field Analysis", https://tvst.arvojournals.org/article.aspx?articleid=2785341)https://tvst.arvojournals.org/article.aspx?articleid=2785341
