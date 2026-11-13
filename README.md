# flipkart-decision-intelligence
Flipkart Decision Intelligence – Machine Learning Capstone Project
## Part 1: Statistical Analysis and Regression (25 Marks)

Objective
The objective of Part 1 is to analyse the Flipkart e-commerce dataset using statistical methods and regression techniques. The analysis covers descriptive statistics, hypothesis testing, sample-size planning, gradient descent, residual analysis, and multiple linear regression.

### R1.1 – Hypothesis Testing and Practical Significance

**Null hypothesis (H0):** The population mean of the selected column is 8.0.
**Calculated z-statistic:** -118.31889879371782
**Critical value:** 1.96
**Decision:** `[H0 was rejected / H0 was not rejected]` because `[explain whether the absolute z_stat exceeds 1.96]`.

Type I Error:** A Type I error in this context would mean concluding that the true average value of the selected Flipkart business metric differs from 8.0 when, in reality, its population mean is 8.0.

The observed difference between the sample mean and the hypothesised value is:

`mean_value - 8.0 = [INSERT GAP] [COLUMN UNITS]`

Although the result is `[statistically significant / not statistically significant]`, the practical importance of the difference depends on its size and its potential impact on business decisions. Statistical significance alone does not establish practical importance.

### R1.2 – Sample Size Planning

**Required sample size (n_required):** `[INSERT n_required]`
**Available observations:** 800

The available dataset contains 800 observations, compared with the calculated required sample size of `[INSERT n_required]`.

**Interpretation:** `[State whether 800 observations are sufficient or insufficient and explain what this means for the planning team when conducting the analysis.]`

### R1.3 – Gradient Descent vs Closed-Form Regression

- **Gradient descent slope:** `[INSERT GD SLOPE]`
- **Closed-form slope:** `[INSERT CLOSED-FORM SLOPE]`
- **gd_slope_gap:** `[INSERT gd_slope_gap]`

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
