# insurance_premium_prediction-project-Machine-learning-

Data Preprocessing and Analysis
In this project, we analyze and preprocess data from an insurance dataset to predict annual premium amounts based on various customer attributes. Below are the steps involved in the data preprocessing and analysis stage.

1. Loading and Inspecting the Data
The dataset (premiums.xlsx) was loaded using the pandas library.
Initial inspection showed the following columns: ['Age', 'Gender', 'Region', 'Marital_status', 'Number Of Dependants', 'BMI_Category', 'Smoking_Status', 'Employment_Status', 'Income_Level', 'Income_Lakhs', 'Medical History', 'Insurance_Plan', 'Annual_Premium_Amount'].
For uniformity, columns were standardized by converting names to lowercase and replacing spaces with underscores.

2. Handling Missing Values and Duplicates
Rows with missing values were removed, and no duplicate rows were found.

3. Outlier Detection and Treatment
Outliers were identified using box plots in the following columns:
![boxplot](https://github.com/user-attachments/assets/d8402e2d-2335-40ef-9a9c-4a9f2d02b9c9)

Number of Dependents: Handled by taking the absolute values to correct any negatives.
Age: Restricted to a range of 18 to 100.
Income Level: Limited to the 99.9th percentile to remove extreme values.
After handling outliers, histograms and scatter plots of numerical columns versus the dependent variable (Annual_Premium_Amount) provided insights into data distribution.

4. Categorical Data Processing
Categorical columns contained a range of values. Here’s a brief overview:
Gender: ['Male', 'Female']
Region: ['Northwest', 'Southeast', 'Northeast', 'Southwest']
Marital Status: ['Unmarried', 'Married']
BMI Category: ['Normal', 'Obesity', 'Overweight', 'Underweight']
Smoking Status: ['No Smoking', 'Regular', 'Occasional', 'Smoking=0', 'Does Not Smoke', 'Not Smoking']
Employment Status: ['Salaried', 'Self-Employed', 'Freelancer']
Income Level: ['<10L', '10L - 25L', '> 40L', '25L - 40L']
Medical History: Includes multiple health conditions.
Insurance Plan: ['Bronze', 'Silver', 'Gold']
Certain inconsistencies in Smoking Status were resolved by grouping values such as {'Not Smoking':'No Smoking', 'Smoking=0':'No Smoking', 'Does Not Smoke':'No Smoking'}.

5. Data Visualization
Distribution of each categorical feature was visualized using bar plots for a quick overview of the data distribution:
![barplot](https://github.com/user-attachments/assets/e422f8a7-a05a-4ddd-bb96-77bde63ec302)

Income Level vs. Insurance Plan: Crosstab analysis showed how insurance plans varied with income levels, visualized with both bar and heatmap plots:
![crosstab](https://github.com/user-attachments/assets/2233fec6-ffad-402c-a55b-4db86d4905ca)

This structured data preprocessing sets a solid foundation for model development, ensuring that the data is clean, consistent, and ready for analysis.

Feature Engineering
To enhance model performance, additional feature engineering steps were applied to the preprocessed data. These steps transformed categorical data, engineered new features, and standardized numerical columns.

1. Medical History Transformation and Risk Score Calculation
The medical_history column includes various diseases, each carrying a distinct risk factor for insurance premiums.
A risk score dictionary was created to assign scores to each disease:

risk_scores = {
    'diabetes': 6,
    'heart disease': 8,
    'high blood pressure': 6,
    'thyroid': 5,
    'no disease': 0,
    'none': 0
}

The medical_history column was split into two separate columns, disease1 and disease2, for cases with multiple conditions, and scores were mapped to these diseases.
The total_risk_score column was created by summing the scores of disease1 and disease2.
The score was then normalized to range between 0 and 1, making it easier for model interpretation:

2. Encoding Categorical Variables
The insurance_plan and income_level columns were mapped to numerical values to capture the inherent ordering:
insurance_plan: {'Bronze': 1, 'Silver': 2, 'Gold': 3}
income_level: {'<10L': 1, '10L - 25L': 2, '25L - 40L': 3, '> 40L': 4}
Nominal categorical variables such as gender, region, marital_status, bmi_category, smoking_status, and employment_status were converted to one-hot encoded columns:

3. Feature Scaling
To standardize features, columns age, number_of_dependants, income_level, income_lakhs, and insurance_plan were scaled using MinMaxScaler to bring them into a range between 0 and 1.

5. Variance Inflation Factor (VIF) Analysis
VIF was calculated to assess multicollinearity among the independent variables, which can negatively impact model stability. Variables with a VIF greater than 10 were considered highly collinear and candidates for removal.
The column income_level was removed based on this analysis to reduce multicollinearity, resulting in a refined feature set.

7. Correlation Analysis
A correlation heatmap was plotted to examine relationships between variables and identify any strong linear dependencies that may affect model performance:

These feature engineering steps prepared the dataset for model training, improving its quality and interpretability.

Model Training
In this section, multiple models were trained to predict the annual premium amount based on the engineered features. Performance was evaluated using metrics like Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) to assess the prediction accuracy.

1. Data Splitting
The dataset was split into training and testing sets, with 30% of the data used for testing

2. Model 1: Linear Regression
A linear regression model was trained as a baseline model.
Scores:
Train score: 0.9282
Test score: 0.9281
Error Metrics:
MSE: 5,165,611.91
RMSE: 2272.80

Feature Importance: From the coefficients, it was observed that insurance_plan, age, and normalised_risk_score significantly contributed to predicting the premium amount:
![feature_linear](https://github.com/user-attachments/assets/23d6490a-5723-4348-aa31-473a155c3b73)

3. Model 2: Ridge Regression
To manage potential overfitting, Ridge regression was applied with an alpha value of 10.
Scores:
Train score: 0.9282
Test score: 0.9280
Ridge regression results were comparable to linear regression, suggesting limited multicollinearity impact.

4. Model 3: XGBoost Regressor
XGBoost regression was employed to capture potential nonlinear relationships in the data.
Scores:
Train score: 0.9861
Test score: 0.9810
Error Metrics:
MSE: 1,367,525.69
RMSE: 1169.41
XGBoost outperformed the linear models with a significantly lower RMSE, indicating higher accuracy.

5. Hyperparameter Tuning
RandomizedSearchCV was used to optimize the XGBoost model’s parameters:
Best parameters: {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1}
Best score: 0.9809

Feature Importance: The tuned XGBoost model confirmed age and insurance_plan as the most influential features:
![feature_xgboost](https://github.com/user-attachments/assets/1978d6bb-d2a0-4cc3-b78c-6565bdee6fc8)

The results from model training indicate that the XGBoost model is the best-performing model in terms of accuracy, making it the preferred model for predicting insurance premiums.

Error Analysis
Error analysis was conducted to identify specific cases where the model's predictions differed significantly from the actual values. The following steps detail the findings and adjustments made:

1. Initial Error Analysis
Predictions (y_pred) were generated using the best model (XGBoost), and residuals (differences between predictions and actual values) were calculated.
The percentage difference (diff_pct) was also computed to understand the relative error.
Initial findings:
30% of the customers had a prediction error greater than 10%, suggesting they were overcharged or undercharged significantly.
549 customers were identified with an error over 50%.
   
2. Distribution of Residuals
A histogram of diff_pct was plotted to examine the distribution of errors, revealing a heavy margin of error in specific data groups, particularly based on the age feature.

3. Focus on Age-Based Errors
Further examination showed that most of the extreme errors occurred in the age group 18-25.
97% quantile of age for extreme errors was 25, indicating that customers under 25 had significantly higher errors.

4. Subset Training: Age Group 18-25
The dataset was split to train a separate model for the age group 18-25, aiming to improve accuracy in this segment.
Linear Regression Results: Train score = 0.602, Test score = 0.605, indicating low predictive power.
XGBoost Results: Train score = 0.725, Test score = 0.564, also yielding low accuracy.
Error Rate: 73% of predictions for age 18-25 customers were off by more than 10%, confirming high error sensitivity within this group.

5. Feature Engineering: Adding Genetic Factor
To address the high error rate, a new feature called genetical factor was introduced, hypothesizing it might impact the premium for younger customers.
After adding this feature, the error rate decreased significantly:
For age group 18-25: Error rate dropped to 2%.
For customers outside this age group: Error rate reduced to 3.2%.

Summary and Insights

Key Findings: Younger customers (18-25) had higher prediction errors, suggesting the initial model did not fully capture their premium determinants.

Adjustments: Adding the genetical factor feature successfully reduced the error rate, making it more acceptable across all age groups.

Remaining Errors: With the genetic factor in place, the error rates of 2% (for 18-25) and 3.2% (for others) were within an acceptable range, minimizing overcharging or undercharging risks.
This refined model now provides improved accuracy by focusing on high-error segments and adding relevant features.

