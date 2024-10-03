


## Step 1: Breast Cancer Dataset Exploration

### Exploratory Visualizations
    
<img width="595" alt="Screenshot 2024-10-03 at 11 50 35" src="https://github.com/user-attachments/assets/4b3b5f9a-7407-477e-8529-29549e8202e5">

  
<img width="625" alt="Screenshot 2024-10-03 at 11 52 23" src="https://github.com/user-attachments/assets/f5b47bbe-6cb5-4610-8119-dfc3ffe63bab">

<img width="612" alt="Screenshot 2024-10-03 at 11 52 32" src="https://github.com/user-attachments/assets/de2bb3ca-c42c-4ed3-9811-b37a390c8013">

<img width="603" alt="Screenshot 2024-10-03 at 11 52 46" src="https://github.com/user-attachments/assets/b58499f8-6ac4-486a-a1aa-3b7cb673d839">

<img width="606" alt="Screenshot 2024-10-03 at 11 53 11" src="https://github.com/user-attachments/assets/baa741e3-8845-4c11-b70c-fdf3b91fe7e3">

<img width="622" alt="Screenshot 2024-10-03 at 11 53 29" src="https://github.com/user-attachments/assets/51fc369f-82e1-4891-a42a-83c1fb0c6b75">


Conclusions from EDA and steps forward

From the plot and from the covariance table, it seems like most of the columns are highly correlated. However, the columns radius_worst', 'radius_mean', 'perimeter_mean', 'perimeter_worst' exhibit the highest degree of collinearity with all other variables.

## Step 2: I Develop a Logistic Regressiom Model to Predict Whether Cancer is Malignant or Benign

<img width="769" alt="Screenshot 2024-10-03 at 11 54 26" src="https://github.com/user-attachments/assets/572020f0-1e67-46e5-9eb0-644e6b4a52ff">

 
# Step 3: I Evaluate the Model:

    The accuracy in my model is: 0.9035087719298246. This means that my model correctly predicts the outcome of 90.35% of the time on my test data set.
    The RMSE in my model is: 0.31063037209869776. This means that my model's prediction are, on average, 0.3106 units away from the actual value.
    The R^2 of my model is: 0.5892564690468391. This means that 58.93% of the variance in my dependent variable is explained by the model. For Linear Regression, a higher R^2 is a better fit, however, since this is a Logistic Regression, the interpretation varies. Overall, I think this is a good model when it comes to its predictability. Since I am not trying to evaluate and improve the accuracy of every single predictor, but its predictive power, I consider this to be a good model when it comes to prediction.

### Focusing on Interactions:
To check how certain variables might interact and influence the diagnosis, I decided to focus on three interactions: 
* 1. Does the diagnosis change based on the mean of the area in relation to mean of the texture?
  2.  Does the diagnosis change based on the effect of the mean of the symmetry in relation to the mean of the compactess mean?
  3. Does the diagnosis change based on the relationship between the radius mean and area mean?
     
### 1. Does the diagnosis change based on the mean of the area in relation to mean of the texture?

<img width="536" alt="Screenshot 2024-10-03 at 11 55 11" src="https://github.com/user-attachments/assets/88c9f2f8-2b98-44a5-a82f-16b22149ba7a">



Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

## Interpretation of this model: Does the diagnosis change the area_mean in relation to texture_mean?

If the diagnosis indicates that the cancer is benign, then when texture_mean increases by one point, the area_mean decreases by 1.3 points, holding all other variables constant. However, this is not statistically significant. The baseline is 273.8978 If the diagnosis indicates that the cancer is malign, then, the baseline is 273.8978+ 486.3076 = 760.2054. For each one-unit increase in texture_mean, the change in area_mean would be: (-1.3127) + (11.4110 * 1) = 10.0983.

###   2.  Does the diagnosis change based on the effect of the mean of the symmetry in relation to the mean of the compactess mean?

<img width="513" alt="Screenshot 2024-10-03 at 11 55 39" src="https://github.com/user-attachments/assets/ba0fdc39-726c-44b8-be82-86868fddad4a">



Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
Interpretation of the model: Does the diagnosis influence the effect of symmetry_mean in relation to its compactness_mean?

If the diagnosis is "benign": By each point increase of symmetry mean, its compactness mean increases by 0.5189. If the diagnosis is "Malign": By each point increase of symmetry mean, its compactness mean changes by 1.3635 (i.e. 0.5189+0.8446). All of these are significant.

 ### 3. Does the diagnosis change based on the relationship between the radius mean and area mean?
 
<img width="505" alt="Screenshot 2024-10-03 at 11 55 55" src="https://github.com/user-attachments/assets/f622cd9a-f95c-4a31-8150-a373c68c66f2">



#### Interpretation of model: does the diagnosis influence the relationship between radius_mean and area_mean?

If the diagnosis is benign: For every increase point in area_mean, its radius_mean increases by 0.0132, with a baseline of 6.0445. If the diagnosis is malign: the baseline changes from 6.0445 to 9.0277 (6.0445 + 2.9832), and for every increase point in area_mean, radius_mean increases by 0.0086 (0.0132 - 0.0046).
issues that occur when using a linear regression model to predict binary outcomes.

IMPORTANT DISCLAIMER: Using linear regression to predict binary outcomes is problematic because it violates the OLS assumptions and can lead to inappropriate output interpretation. Specifically, linear regression can predict values outside the 0-1 range.
This conflicts with the binary nature of the outcomes. Moreover, it assumes constant variance and a linear relationship between predictors and the outcome, which is often not the case in binary data, leading to inefficiencies and inaccuracies.

Finally, the normality assumption for residuals doesn't hold for binary outcomes, and standard model evaluation metrics like R² do not accurately reflect the explanatory power of the model.


## Step 3: I create a reduced model with only a subset of significant predictors. To do that, I used the FDR Cutoff method

<img width="598" alt="Screenshot 2024-10-03 at 11 57 04" src="https://github.com/user-attachments/assets/e0bff080-6b4e-47d0-a2c9-5d20cba61294">

<img width="610" alt="Screenshot 2024-10-03 at 11 57 21" src="https://github.com/user-attachments/assets/db4a476a-d402-4c15-9fc5-ab75dfcb55cb">


### Evaluation and Conclusion:

Full Model:
Mean R² in Cross-Validation: 0.7274 R² Scores Across Folds: Range from a low of 0.6232 to a high of 0.8156.

The full model shows relatively high and consistent R² values across the 10 folds, suggesting good model performance and stability. The variation in R² scores across the folds is present but not overly large, indicating that the model is reasonably robust to the specifics of the fold data. The mean R² in cross-validation (0.7274) is quite close to the R² obtained on the entire training set (0.7800), suggesting that the model is generalizing well to unseen data.

Reduced Model:
Mean R² in Cross-Validation: 0.2263 R² Scores Across Folds: Range from a low of 0.0913 to a high of 0.3755.

The reduced model shows much lower R² values compared to the full model, with a considerable range in performance across the folds. The inconsistency in R² scores (varying from very low to moderate) across folds indicates that the reduced model's performance is quite sensitive to the specific data in each fold. The mean R² in cross-validation (0.2263) is significantly lower than the R² on the entire training set (0.4959). This might imply that the reduced model is not capturing enough complexity in the data to generalize well.
