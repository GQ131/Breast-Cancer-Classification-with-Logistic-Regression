# Breast Cancer Detection Project

## Critical question: 
How can we develop an efficient and reliable system to support healthcare professionals in accurately diagnosing breast cancer?

## Project Overview
This project addresses this critical question by leveraging logistic regression to build a predictive model that distinguishes between malignant and benign cases, providing a foundational approach for implementing AI in healthcare diagnostics.

The dataset contains features computed from a digitized image of a fine needle aspirate of a breast mass, such as texture, area, radius, and symmetry. 



## Step 1:
The first step involves understanding the dataset through visualization and statistical exploration. Below are some key visualizations used for analysis:

#### Correlation Heatmap:
        Visualizes how each feature relates to others.
        Highlights that several features are highly correlated, such as radius_mean, radius_worst, perimeter_mean, and perimeter_worst.
<img width="622" alt="Screenshot 2024-10-03 at 11 53 29" src="https://github.com/user-attachments/assets/51fc369f-82e1-4891-a42a-83c1fb0c6b75">

#### Scatter Plots and Pair Plots:
        Helped identify relationships between pairs of variables.
        Provided insights into how variables cluster for malignant and benign classifications.
<img width="595" alt="Screenshot 2024-10-03 at 11 50 35" src="https://github.com/user-attachments/assets/4b3b5f9a-7407-477e-8529-29549e8202e5">

<img width="625" alt="Screenshot 2024-10-03 at 11 52 23" src="https://github.com/user-attachments/assets/f5b47bbe-6cb5-4610-8119-dfc3ffe63bab">

<img width="612" alt="Screenshot 2024-10-03 at 11 52 32" src="https://github.com/user-attachments/assets/de2bb3ca-c42c-4ed3-9811-b37a390c8013">

<img width="606" alt="Screenshot 2024-10-03 at 11 53 11" src="https://github.com/user-attachments/assets/baa741e3-8845-4c11-b70c-fdf3b91fe7e3">

    
#### Box Plots:
Showed the distribution of key features, making it easier to detect outliers and differences between malignant and benign cases.


<img width="603" alt="Screenshot 2024-10-03 at 11 52 46" src="https://github.com/user-attachments/assets/b58499f8-6ac4-486a-a1aa-3b7cb673d839">



### Key Findings from EDA:

    Many columns exhibit multicollinearity, which might affect model stability.
    Specific features such as radius_worst, radius_mean, perimeter_mean, and perimeter_worst display the highest correlation levels, indicating redundancy in some predictive variables.


## Step 2:Building a Logistic Regression Model
Model Overview:

<img width="769" alt="Screenshot 2024-10-03 at 11 54 26" src="https://github.com/user-attachments/assets/572020f0-1e67-46e5-9eb0-644e6b4a52ff">



I used a Logistic Regression model to predict whether the cancer is malignant or benign. This model is well-suited for binary classification tasks like this one. The dataset was split into training and test sets, and standard preprocessing steps (e.g., scaling and feature selection) were applied.
Model Performance:

    Accuracy: 90.35% on the test dataset.
    RMSE (Root Mean Square Error): 0.3106.
    R² (R-squared): 0.589, indicating that 58.93% of the variability in cancer diagnosis is explained by the model. However, since this is a logistic regression model, R² should be interpreted cautiously.

Model Limitations:

Using linear regression for binary outcomes (e.g., benign/malignant) has drawbacks, as it can violate key assumptions like constant variance and normality. As a result, logistic regression is a better fit for this task due to its probabilistic approach and capability to handle binary outputs.
Step 3: Interaction Analysis
Exploring Interactions:

## Step 3: Interaction Analysis
Exploring Interactions:

I explored how certain variables interact to understand their combined effect on cancer diagnosis. Here are the three main interactions evaluated:

    Effect of area_mean in Relation to texture_mean:
    
<img width="536" alt="Screenshot 2024-10-03 at 11 55 11" src="https://github.com/user-attachments/assets/88c9f2f8-2b98-44a5-a82f-16b22149ba7a">

        When texture_mean increases by one unit, the area_mean decreases by 1.3 units if the cancer is benign.
        If the cancer is malignant, area_mean increases by 10.1 units for every unit increase in texture_mean.

    Effect of symmetry_mean in Relation to compactness_mean:
<img width="513" alt="Screenshot 2024-10-03 at 11 55 39" src="https://github.com/user-attachments/assets/ba0fdc39-726c-44b8-be82-86868fddad4a">

        For benign cases, compactness_mean increases by 0.5189 for each unit increase in symmetry_mean.
        For malignant cases, the increase is higher at 1.3635 units.

    Effect of area_mean in Relation to radius_mean:
<img width="505" alt="Screenshot 2024-10-03 at 11 55 55" src="https://github.com/user-attachments/assets/f622cd9a-f95c-4a31-8150-a373c68c66f2">
        If the cancer is benign, radius_mean increases by 0.0132 for every unit increase in area_mean, starting from a baseline of 6.0445.
        If the cancer is malignant, the increase is 0.0086 units, starting from a higher baseline of 9.0277.

Important Consideration:

Using linear regression to predict binary outcomes can lead to inappropriate interpretations, as it may predict probabilities outside the 0-1 range. Logistic regression, which models probabilities directly, is more appropriate for this kind of classification problem.



## Step 4: Feature Selection with FDR Cutoff Method
Creating a Reduced Model:

<img width="598" alt="Screenshot 2024-10-03 at 11 57 04" src="https://github.com/user-attachments/assets/e0bff080-6b4e-47d0-a2c9-5d20cba61294">

<img width="610" alt="Screenshot 2024-10-03 at 11 57 21" src="https://github.com/user-attachments/assets/db4a476a-d402-4c15-9fc5-ab75dfcb55cb">

After initial analysis, I reduced the number of features using the False Discovery Rate (FDR) Cutoff method to include only the most significant predictors.

    Full Model Performance:
        Mean R² in Cross-Validation: 0.7274
        R² Scores Across Folds: Ranged from 0.6232 to 0.8156, indicating high and consistent performance.

    Reduced Model Performance:
        Mean R² in Cross-Validation: 0.2263
        R² Scores Across Folds: Ranged from 0.0913 to 0.3755, showing a significant drop in predictability and robustness.

Model Conclusion:

The reduced model struggles to capture complexity and nuance in the dataset compared to the full model, suggesting that many of the removed features were indeed valuable in predicting outcomes.
