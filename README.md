# USA-Visa-Approval-Prediction

# Problem Statement:
The Office of Foreign Labor Certification (OFLC) reviews job certification applications from employers seeking to bring foreign workers to the United States. Due to the large volume of applications in recent years, OFLC requires a machine learning model to help shortlist visa applicants based on previous data.

This project aims to build a classification model to predict whether a visa application will be approved or denied. The model will help recommend suitable profiles for visa certification or denial based on specific criteria that influence the decision.

# Objective:
Develop a machine learning classification model to predict the approval status of US visa applications. This model can be used to assist the OFLC in efficiently processing visa certifications.

# Dataset:
Data Collection.
The Dataset is part of Office of Foreign Labor Certification (OFLC)
The data consists of 25480 Rows and 12 Columns
https://www.kaggle.com/datasets/moro23/easyvisa-dataset

The dataset contains various features relevant to the visa application process:
Columns:
- case_id: Unique ID for each application
- continent: Applicant's continent
education_of_employee: Education level of the applicant
has_job_experience: Whether the applicant has job experience (Yes/No)
requires_job_training: Whether the job requires training (Yes/No)
no_of_employees: Number of employees in the company
yr_of_estab: Year the company was established
region_of_employment: Region where the applicant will be employed
prevailing_wage: Wage offered to the applicant
unit_of_wage: Unit in which the wage is measured (e.g., yearly, monthly)
full_time_position: Whether the position is full-time (Yes/No)
case_status: Target variable indicating whether the visa was approved or denied


# Tech Stack:
Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Chi2, etc

# Approach:
* Data Preprocessing:
Handled missing values, if any.
Performed a multicollinearity check using the Chi-squared (chi2) test to identify highly correlated features and remove those that did not contribute independently to the model’s performance.
Converted categorical features (like continent, education_of_employee, has_job_experience, etc.) into numerical representations using one-hot encoding or label encoding.
Scaled numeric features such as no_of_employees, yr_of_estab, and prevailing_wage to ensure consistency in model training.
Power Transformation: Used power transformation to make features like prevailing_wage and no_of_employees more Gaussian-like for better model performance.

* Exploratory Data Analysis (EDA):
Analyzed trends in visa approval rates by continent, education level, and job experience, etc
Visualized the distribution of wages, employment, regions, etc to identify key factors influencing visa approvals.

* Visualizations:
Heatmaps for correlation
Bar plots and histograms for feature distribution
Box plots to check for outliers

* Model Selection:
Built multiple machine learning models, including:
Logistic Regression
Decision Tree
Random Forest
XGBoost, Adaboost, etc 
Applied cross-validation to assess the generalization of the models.

* Evaluation:
Evaluated the models using metrics like accuracy, precision, recall, and F1-score to account for class imbalance (approved vs. denied cases).
Visualized the results using confusion matrices and ROC-AUC curves.

* Results:
Best Model: # Best Model is K-Nearest Neighbor(KNN) with Accuracy 97%. The model performed well in distinguishing between approved and denied visa applications.

* Conclusions:
The model identified that features like job experience, education level, and prevailing wage were significant factors in determining visa approval.
This model can be integrated into a decision support system for OFLC to recommend visa approval based on historical data.
Future improvements could include using additional external data like employer reputation and visa policy changes.

