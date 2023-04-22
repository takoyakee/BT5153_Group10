# Stay or Stray: Machine Learning Decodes the Attrition Mystery
This repository provides the source codes used in EDA, Model Training and Tuning, Feature Selection, Global and Local Interpretability analysis for the problem of attrition prediction as part of BT5153.

The final model aids organisation in predicting attrition that maximises expected value according to needs and provides interpretable and actioable insights for management.

The dataset is retrieved from Kaggle: [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) 

# Outline of notebook:
- Impots: pip install, import library, loading data
- EDA: Exploratory Data Analysis (general processing, correlations, uni/multi variate analysis)
- Data Processing: Dropping features, Feature Engineering, Creating Data Pipeline for ML
- Main model + helper functions: All functions used for training, tuning, selection and feature importance
- Baseline: Defines 3 cost-benefit matrix (CBM) + EV for three baseline strategies (Random, No Target, Target All)
- Model Selection for CBM1 (Train+Tune, Evaluate+Choose Model, Feature Selection, Optimising EV through threshold)
- Model Selection for CBM2
- Model Selection for CBM3
- Results 
- Lime Example for Display (LIME analysis + [Dashboard for HR use](https://attrition-analytics.herokuapp.com/) )

# Brief explanation + function locations:
- EDA: Bivariate Analysis, Multivariate Analysis, Correlation (EDA)
- Model Training and Tuning: 5 folds CV for KNN, LR, SVM, DT, GB, RF (Main model + helper functions, )
- Feature Selection: Recursive Feature Elimination with CV (Main model + helper functions)
- Global Interpretability: RF feature impt, Permutation feature impt (Main model + helper functions)
- Local Interpretability: LIME + processed LIME for dashboard (Lime Example for Display)
