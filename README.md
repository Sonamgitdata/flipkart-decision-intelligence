# flipkart-decision-intelligence
Flipkart Decision Intelligence – Machine Learning Capstone Project
## Part 1: Statistical Analysis and Regression (25 Marks)

Objective
The objective of Part 1 is to analyse the Flipkart e-commerce dataset using statistical methods and regression techniques. The analysis covers descriptive statistics, hypothesis testing, sample-size planning, gradient descent, residual analysis, and multiple linear regression.
### R1.1 – Hypothesis Testing and Practical Significance

**Null hypothesis (H0):** The population mean of the selected column is 8.0.
**Calculated z-statistic:** -118.31889879371782
**Critical value:** 1.96
**Decision:** If abs(z_stat) > 1.96, reject H0.
              If abs(z_stat) ≤ 1.96, do not reject H0.

Type I Error:** A Type I error in this context would mean concluding that the true average value of the selected Flipkart business metric differs from 8.0 when, in reality, its population mean is 8.0.

The observed difference between the sample mean and the hypothesised value is:

mean_value - 8.0 = 32.249375 - 8.0 = 24.249375

The sample mean is 24.249375 units higher than the hypothesised population mean of 8.0. The result is statistically significant because the calculated absolute z-statistic is approximately 42.35, which is considerably greater than the critical value of 1.96. The 95% confidence interval (31.1254, 33.3734) also does not contain the hypothesised value of 8.0.

Although the difference is statistically significant, its practical importance depends on the business context and the units of the selected column. A difference of 24.249375 units may represent a meaningful business impact, but statistical significance alone does not establish practical importance.

### R1.2 – Sample Size Planning

**Required sample size (n_required):** 62
**Available observations:** 800

The available dataset contains 800 observations, compared with the calculated required sample size of 62.

**Interpretation:** The available dataset of 800 observations is sufficient for the analysis, as it significantly exceeds the calculated required sample size of 62. This means the dataset provides more than enough observations to meet the minimum sample size requirement, supporting reliable statistical analysis and potentially improving the precision of the results. The planning team can proceed with the analysis using the available 800 observations, provided that the data quality, representativeness, and assumptions of the selected statistical methods are satisfied.

### R1.3 – Gradient Descent vs Closed-Form Regression

**Gradient descent slope:**  -2.412169083785412
**Closed-form slope:** -0.14903661811876878
**gd_slope_gap:** = |Gradient descent slope − Closed-form slope|

= |-2.412169083785412 − (-0.14903661811876878)|= 2.263132465666643

The small difference between the gradient descent and closed-form slopes indicates that gradient descent successfully converged towards the optimal linear regression solution. Both approaches produce approximately the same fitted line, demonstrating that the iterative optimisation method reached the analytical solution.

### R1.4 – Residual Analysis: Raw vs Log Model

- **resid_sd_ratio_raw:** `[INSERT VALUE]`
- **resid_sd_ratio_log:** `[INSERT VALUE]`

The raw regression model `[shows / does not show]` a funnel-shaped residual pattern, where the spread of residuals `[increases / does not increase]` as the fitted values increase.

The log-transformed model `[removes / does not remove]` this funnel pattern. This suggests that the log transformation `[improves / does not sufficiently improve]` the consistency of residual variance.

### R1.5 – Multiple Regression and Adjusted R-Squared Comparison

**Predictors with absolute coefficients greater than 0.05 (excluding the constant):**

- `[PREDICTOR 1]`: `[COEFFICIENT]`
- `[PREDICTOR 2]`: `[COEFFICIENT]`
- `[PREDICTOR 3]`: `[COEFFICIENT]`

Apply the absolute coefficient threshold strictly: include every predictor satisfying `|coefficient| > 0.05` and exclude all others.

**Model comparison:**

- **mr_adj_r2:** `[INSERT VALUE]`
- **loglog_adj_r2:** `[INSERT VALUE]`

The adjusted R-squared values indicate whether the additional predictors improve the explanatory performance of the multiple regression model compared with the log-log model.

`[Explain which model has the higher adjusted R-squared and whether the improvement justifies including the additional predictors.]`

The coefficient threshold is a mechanical selection rule and should not be interpreted as a ranking of predictor importance. Coefficients have different measurement units, so their magnitudes are not directly comparable.

### Part 1 Conclusion

Part 1 applies statistical testing and regression techniques to understand the dataset, evaluate differences from a hypothesised mean, determine sample-size requirements, compare optimisation methods, investigate residual behaviour, and assess regression model performance.

The results provide a statistical foundation for the subsequent classification, clustering, and machine-learning tasks in this capstone project.
