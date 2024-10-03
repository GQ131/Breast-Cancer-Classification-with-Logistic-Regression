


## Step 1: I explore the Breast Cancer Dataset via visualizations
    <img width="595" alt="Screenshot 2024-10-03 at 11 50 35" src="https://github.com/user-attachments/assets/4b3b5f9a-7407-477e-8529-29549e8202e5">

  


Conclusions from EDA and steps forward

From the plot and from the covariance table, it seems like most of the columns are highly correlated. However, the columns radius_worst', 'radius_mean', 'perimeter_mean', 'perimeter_worst' exhibit the highest degree of collinearity with all other variables.

## Step 2: I Develop a Logistic Regressiom Model to Predict Whether Cancer is Malignant or Benign

                 Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:              diagnosis   No. Observations:                  569
Model:                            GLM   Df Residuals:                      542
Model Family:                Binomial   Df Model:                           26
Link Function:                  Logit   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -15.603
Date:                Thu, 25 Jan 2024   Deviance:                       31.207
Time:                        09:59:15   Pearson chi2:                     634.
No. Iterations:                    14   Pseudo R-squ. (CS):             0.7180
Covariance Type:            nonrobust                                         
===========================================================================================
                              coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------------------
const                    -117.2788     48.957     -2.396      0.017    -213.232     -21.325
texture_mean                0.0908      0.557      0.163      0.870      -1.000       1.182
area_mean                  -0.0521      0.035     -1.492      0.136      -0.120       0.016
smoothness_mean           247.2708    191.430      1.292      0.196    -127.924     622.466
compactness_mean         -310.2744    169.921     -1.826      0.068    -643.314      22.765
concavity_mean            191.0264    115.813      1.649      0.099     -35.964     418.016
concave points_mean       152.7277    164.460      0.929      0.353    -169.607     475.063
symmetry_mean             -89.6497     67.807     -1.322      0.186    -222.549      43.250
fractal_dimension_mean    249.7130    369.607      0.676      0.499    -474.703     974.129
radius_se                  30.3750     35.383      0.858      0.391     -38.974      99.724
texture_se                 -5.4749      4.081     -1.341      0.180     -13.474       2.524
perimeter_se               -2.3699      2.533     -0.936      0.349      -7.334       2.594
area_se                     0.1565      0.380      0.412      0.680      -0.588       0.901
smoothness_se             374.9212    483.483      0.775      0.438    -572.689    1322.531
compactness_se            404.4846    259.223      1.560      0.119    -103.583     912.552
concavity_se             -308.9263    165.758     -1.864      0.062    -633.806      15.953
concave points_se        1704.0735    896.326      1.901      0.057     -52.693    3460.840
symmetry_se              -311.2392    287.686     -1.082      0.279    -875.094     252.616
fractal_dimension_se    -5457.7953   2702.566     -2.019      0.043   -1.08e+04    -160.863
texture_worst               0.9554      0.554      1.724      0.085      -0.131       2.041
area_worst                  0.0620      0.034      1.805      0.071      -0.005       0.129
smoothness_worst          -56.2854    116.769     -0.482      0.630    -285.148     172.577
compactness_worst         -18.6801     41.431     -0.451      0.652     -99.884      62.523
concavity_worst            14.5665     27.162      0.536      0.592     -38.670      67.803
concave points_worst       -3.6443     72.059     -0.051      0.960    -144.877     137.588
symmetry_worst             76.3102     49.933      1.528      0.126     -21.556     174.177
fractal_dimension_worst   528.5356    283.225      1.866      0.062     -26.576    1083.647
===========================================================================================

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

OLS Regression Results Dep. Variable: 	area_mean 	R-squared: 	0.507
Model: 	OLS 	Adj. R-squared: 	0.505
Method: 	Least Squares 	F-statistic: 	193.8
Date: 	Thu, 25 Jan 2024 	Prob (F-statistic): 	2.09e-86
Time: 	09:59:15 	Log-Likelihood: 	-3941.8
No. Observations: 	569 	AIC: 	7892.
Df Residuals: 	565 	BIC: 	7909.
Df Model: 	3 		
Covariance Type: 	nonrobust 		
	coef 	std err 	t 	P>|t| 	[0.025 	0.975]
Intercept 	486.3076 	60.312 	8.063 	0.000 	367.845 	604.770
texture_mean 	-1.3127 	3.286 	-0.399 	0.690 	-7.767 	5.142
diagnosis 	273.8978 	115.885 	2.364 	0.018 	46.279 	501.516
texture_mean:diagnosis 	11.4110 	5.582 	2.044 	0.041 	0.447 	22.375
Omnibus: 	196.390 	Durbin-Watson: 	1.858
Prob(Omnibus): 	0.000 	Jarque-Bera (JB): 	1207.511
Skew: 	1.382 	Prob(JB): 	6.20e-263
Kurtosis: 	9.579 	Cond. No. 	259.


Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

## Interpretation of this model: Does the diagnosis change the area_mean in relation to texture_mean?

If the diagnosis indicates that the cancer is benign, then when texture_mean increases by one point, the area_mean decreases by 1.3 points, holding all other variables constant. However, this is not statistically significant. The baseline is 273.8978 If the diagnosis indicates that the cancer is malign, then, the baseline is 273.8978+ 486.3076 = 760.2054. For each one-unit increase in texture_mean, the change in area_mean would be: (-1.3127) + (11.4110 * 1) = 10.0983.

###   2.  Does the diagnosis change based on the effect of the mean of the symmetry in relation to the mean of the compactess mean?


OLS Regression Results Dep. Variable: 	compactness_mean 	R-squared: 	0.582
Model: 	OLS 	Adj. R-squared: 	0.580
Method: 	Least Squares 	F-statistic: 	262.5
Date: 	Thu, 25 Jan 2024 	Prob (F-statistic): 	1.18e-106
Time: 	09:59:15 	Log-Likelihood: 	1114.9
No. Observations: 	569 	AIC: 	-2222.
Df Residuals: 	565 	BIC: 	-2204.
Df Model: 	3 		
Covariance Type: 	nonrobust 		
	coef 	std err 	t 	P>|t| 	[0.025 	0.975]
Intercept 	-0.0103 	0.013 	-0.801 	0.423 	-0.036 	0.015
symmetry_mean 	0.5189 	0.073 	7.097 	0.000 	0.375 	0.663
diagnosis 	-0.1076 	0.021 	-5.119 	0.000 	-0.149 	-0.066
symmetry_mean:diagnosis 	0.8446 	0.112 	7.520 	0.000 	0.624 	1.065
Omnibus: 	53.576 	Durbin-Watson: 	1.817
Prob(Omnibus): 	0.000 	Jarque-Bera (JB): 	74.812
Skew: 	0.698 	Prob(JB): 	5.69e-17
Kurtosis: 	4.099 	Cond. No. 	97.0


Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
Interpretation of the model: Does the diagnosis influence the effect of symmetry_mean in relation to its compactness_mean?

If the diagnosis is "benign": By each point increase of symmetry mean, its compactness mean increases by 0.5189. If the diagnosis is "Malign": By each point increase of symmetry mean, its compactness mean changes by 1.3635 (i.e. 0.5189+0.8446). All of these are significant.

 ### 3. Does the diagnosis change based on the relationship between the radius mean and area mean?

OLS Regression Results Dep. Variable: 	radius_mean 	R-squared: 	0.992
Model: 	OLS 	Adj. R-squared: 	0.992
Method: 	Least Squares 	F-statistic: 	2.384e+04
Date: 	Thu, 25 Jan 2024 	Prob (F-statistic): 	0.00
Time: 	09:59:15 	Log-Likelihood: 	-144.14
No. Observations: 	569 	AIC: 	296.3
Df Residuals: 	565 	BIC: 	313.7
Df Model: 	3 		
Covariance Type: 	nonrobust 		
	coef 	std err 	t 	P>|t| 	[0.025 	0.975]
Intercept 	6.0445 	0.059 	101.606 	0.000 	5.928 	6.161
area_mean 	0.0132 	0.000 	106.791 	0.000 	0.013 	0.013
diagnosis 	2.9832 	0.085 	34.963 	0.000 	2.816 	3.151
area_mean:diagnosis 	-0.0046 	0.000 	-33.399 	0.000 	-0.005 	-0.004
Omnibus: 	486.437 	Durbin-Watson: 	2.038
Prob(Omnibus): 	0.000 	Jarque-Bera (JB): 	17486.287
Skew: 	-3.539 	Prob(JB): 	0.00
Kurtosis: 	29.220 	Cond. No. 	6.99e+03



#### Interpretation of model: does the diagnosis influence the relationship between radius_mean and area_mean?

If the diagnosis is benign: For every increase point in area_mean, its radius_mean increases by 0.0132, with a baseline of 6.0445. If the diagnosis is malign: the baseline changes from 6.0445 to 9.0277 (6.0445 + 2.9832), and for every increase point in area_mean, radius_mean increases by 0.0086 (0.0132 - 0.0046).
issues that occur when using a linear regression model to predict binary outcomes.

IMPORTANT DISCLAIMER: Using linear regression to predict binary outcomes is problematic because it violates the OLS assumptions and can lead to inappropriate output interpretation. Specifically, linear regression can predict values outside the 0-1 range.
This conflicts with the binary nature of the outcomes. Moreover, it assumes constant variance and a linear relationship between predictors and the outcome, which is often not the case in binary data, leading to inefficiencies and inaccuracies.

Finally, the normality assumption for residuals doesn't hold for binary outcomes, and standard model evaluation metrics like R² do not accurately reflect the explanatory power of the model.


## Step 3: I create a reduced model with only a subset of significant predictors. To do that, I used the FDR Cutoff method


### Evaluation and Conclusion:

Full Model:
Mean R² in Cross-Validation: 0.7274 R² Scores Across Folds: Range from a low of 0.6232 to a high of 0.8156.

The full model shows relatively high and consistent R² values across the 10 folds, suggesting good model performance and stability. The variation in R² scores across the folds is present but not overly large, indicating that the model is reasonably robust to the specifics of the fold data. The mean R² in cross-validation (0.7274) is quite close to the R² obtained on the entire training set (0.7800), suggesting that the model is generalizing well to unseen data.

Reduced Model:
Mean R² in Cross-Validation: 0.2263 R² Scores Across Folds: Range from a low of 0.0913 to a high of 0.3755.

The reduced model shows much lower R² values compared to the full model, with a considerable range in performance across the folds. The inconsistency in R² scores (varying from very low to moderate) across folds indicates that the reduced model's performance is quite sensitive to the specific data in each fold. The mean R² in cross-validation (0.2263) is significantly lower than the R² on the entire training set (0.4959). This might imply that the reduced model is not capturing enough complexity in the data to generalize well.
