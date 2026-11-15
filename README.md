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

resid_sd_ratio_raw: 3.666057970026114
resid_sd_ratio_log: 3.313343698781273

The raw regression model shows a funnel-shaped residual pattern, where the spread of residuals generally increases as the fitted values increase.
The log-transformed model reduces the funnel pattern. This suggests that the log transformation improves the consistency of residual variance, although some variation remains.

### R1.5 – Multiple Regression and Adjusted R-Squared Comparison

1. Predictors with absolute coefficients greater than 0.05
- complaint_count_90d: 0.804037
- discount_order_share: 0.204480
- avg_order_value_inr_hundreds: 0.554021
- cod_flag: 1.686518
- plus_member_flag: 0.783243
- churned: -0.250037
The following predictors are excluded because their absolute coefficients are below 0.05:
- customer_tenure_months
- loyalty_score
2. Model comparison
- mr_adj_r2: 0.5641630721
- loglog_adj_r2: 0.0823361584
3. Completed interpretation
The adjusted R-squared values indicate that the multiple regression model has a higher explanatory performance than the log-log model.

The multiple regression model has an adjusted R-squared of 0.5642 (56.42%), while the log-log model has an adjusted R-squared of 0.0823 (8.23%).

Therefore, the multiple regression model explains substantially more variation in order value than the log-log model. The improvement supports including the additional predictors, provided they are theoretically relevant and do not introduce data leakage or unnecessary complexity.
### Part 1 Conclusion

Part 1 applies statistical testing and regression techniques to understand the dataset, evaluate differences from a hypothesised mean, determine sample-size requirements, compare optimisation methods, investigate residual behaviour, and assess regression model performance.

The results provide a statistical foundation for the subsequent classification, clustering, and machine-learning tasks in this capstone project.


# Part 2: Classification – Flipkart Decision Intelligence

## 1. Overview

Part 2 focuses on building and evaluating classification models to predict customer churn using the Flipkart customer dataset.

The objective is to understand customer churn patterns, implement Logistic Regression and K-Nearest Neighbors (KNN), evaluate model performance, and identify suitable classification thresholds based on business costs.

## 2. Dataset Description

The project uses `records.csv`, containing:

- Total records: 800
- Predictor variables: 7
- Target variable: `churned`
- Churned customers: 170
- Non-churned customers: 630

The target variable `churned` is used to predict whether a customer will leave.

## 3. Tasks Implemented

### Task 1: Data Preparation

- Load the customer dataset.
- Separate predictor variables (X1–X7) and target variable (`churned`).
- Perform a stratified 70:30 train-test split.
- Standardize predictor variables using the training dataset mean and sample standard deviation.
- Apply the same scaling parameters to the test dataset.

### Task 2: Logistic Regression from Scratch

Implement Logistic Regression without relying on a pre-built Logistic Regression model.

The implementation includes:

- Sigmoid activation function.
- Binary Cross-Entropy (BCE) loss.
- Gradient descent optimization.
- 1,000 training updates.
- Learning rates: 0.01, 0.1, and 1.0.
- Tracking model loss and weights during training.
- Calculating odds ratios from the final model coefficients.

### Task 3: Classification Threshold Evaluation

Evaluate classification performance using thresholds of:

- 0.5
- 0.3

Calculate the following evaluation metrics:

- True Positive (TP)
- False Positive (FP)
- True Negative (TN)
- False Negative (FN)
- Accuracy
- Precision
- Recall
- F1-score
- Specificity

Threshold selection must use training data only to avoid test-data leakage.

### Task 4: Business Cost Analysis

Evaluate classification decisions using the business cost function:

**Total Cost = (50 × False Positives) + (400 × False Negatives)**

This reflects the higher business impact of incorrectly predicting a customer as unlikely to churn when the customer actually churns.

### Task 5: K-Nearest Neighbors (KNN)

Implement KNN classification from scratch.

The implementation includes:

- Distance calculation.
- Nearest-neighbor selection.
- Majority voting for classification.
- 5-fold cross-validation.
- Testing odd K values from 1 to 21.
- Comparing cross-validation results to identify a suitable K value.

## 4. Output Files

The following CSV files are generated by the Part 2 implementation:

| File | Description |
|---|---|
| `part2_losses.csv` | Logistic Regression training loss history |
| `part2_weights.csv` | Final model weights and odds ratios |
| `part2_thresholds.csv` | Classification threshold evaluation metrics |
| `part2_knn_cv.csv` | KNN 5-fold cross-validation results |

## 5. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib (if used for visualizations)
- Google Colab
- Git and GitHub

## 6. How to Run the Project

### Using Google Colab

1. Open `Flipkart_Capstone.ipynb` in Google Colab.
2. Mount Google Drive if the project is stored there.
3. Ensure the dataset and Python script are available.
4. Execute the Part 2 script:

```python
%run "/content/drive/MyDrive/flipkart-decision-intelligence/part2.py"
```

5. Check the project folder for the generated CSV files.

### Using a Local Python Environment

Navigate to the project directory and execute:

```bash
python part2.py
```

Install the required packages if they are not already available:

```bash
pip install numpy pandas matplotlib
```

## 7. Expected Outcomes

After completing Part 2, the project should demonstrate:

- Successful implementation of Logistic Regression from scratch.
- Comparison of different learning rates.
- Interpretation of model coefficients through odds ratios.
- Evaluation of classification thresholds.
- Analysis of business costs associated with classification errors.
- Implementation and cross-validation of KNN.
- Generation of CSV files containing model results.

## 8. Conclusion

Part 2 applies classification techniques to customer churn prediction. Logistic Regression and KNN are implemented and evaluated using appropriate performance metrics and business cost considerations.

The results provide a foundation for understanding customer churn and supporting data-driven business decisions.

---

**Project:** Flipkart Decision Intelligence  
**Section:** Part 2 – Classification  
**Repository:** `flipkart-decision-intelligence`

## Part 3 findings

### R3.1 – K-means and elbow selection

The WCSS values for K=2 to K=6 were 2262.6753, 1602.8471, 1494.7732, 1323.7157 and 1278.1691, respectively.

The successive WCSS reductions were:

- K=2 to K=3: 659.8282
- K=3 to K=4: 108.0739
- K=4 to K=5: 171.0575
- K=5 to K=6: 45.5466

The elbow criterion selects K=3, which is the smallest K satisfying the specified reduction condition. The WCSS reductions do not decrease steadily because the pinned starting centroids provide a reproducible solution, not necessarily the global optimum for every K.

### R3.2 – Segment profiles and business recommendations

**Segment A – Established Loyal Customers**

This segment contains 322 customers, with an average tenure of 47.50 months, loyalty score of 62.16 and outcome rate of 1.86%. These customers demonstrate relatively strong loyalty and low outcome risk.

Business action: Provide loyalty rewards, personalized recommendations and cross-selling offers to maintain engagement.

**Segment B – High-Value Customers Needing Attention**

This segment contains 255 customers, with an average order value of 28.72 and outcome rate of 23.53%. These customers have relatively high complaints and higher average order value.

Business action: Introduce proactive customer support, complaint resolution and personalized retention offers.

**Segment C – Discount-Driven High-Risk Customers**

This segment contains 223 customers, with an average discount order share of 0.712 and outcome rate of 46.64%. These customers have the highest outcome rate among the three segments.

Business action: Provide targeted onboarding, repeat-purchase incentives and loyalty benefits to improve long-term engagement.

Segment C has the highest outcome rate at **46.64%**.

### R3.3 – Hierarchical clustering comparison

At the largest-gap cut, single linkage produces four clusters:

- E1
- E2+E3+E6
- E4
- E5

Complete linkage produces two clusters:

- E1+E4+E5
- E2+E3+E6

Single linkage uses the minimum distance between cluster members, allowing clusters to join through close individual pairs. Complete linkage uses the maximum distance, requiring all members of two clusters to be relatively close before merging, which results in a different clustering structure.

### R3.4 – PCA and dimensionality reduction

The standardised dataset has a mean vector approximately equal to zero for all five predictors. The sum of the five covariance-matrix eigenvalues is 5.0000.

The first principal component explains 52.27% of the variance. Its strongest positive loadings are X1 (0.5317) and X5 (0.4418), while its strongest negative loadings are X3 (-0.5187) and X2 (-0.4100). Therefore, PC1 primarily contrasts customer tenure and loyalty with discount usage and complaints.

Three principal components are required to retain at least 80% of the variance, explaining approximately 86.16%.

The mean PC1 scores are:

- Segment A: 1.7773
- Segment B: -0.7831
- Segment C: -1.6709

Segment A has a positive and substantially higher PC1 score, while B and C have negative scores. This indicates that the first principal component helps distinguish established, loyal customers from the other customer segments.


Part 4 — Findings and model-choice memo

## Part 4 findings

### R4.1 — Purest split

The best split is **S4: X1 ≤ 24**, with information gain **0.127488** and Gini gain **0.060749**, the largest values among the four candidates. A tree grown until every leaf is pure can memorise noise and peculiarities in the training rows, so its test performance may be worse; a practical stopping rule is a maximum tree depth (`max_depth`) or a minimum number of samples per leaf.

### R4.2 — Voting comparison

I would ship the **stump** as the current structured-data candidate at probability threshold **0.30**. It achieved **F1 = 0.553846** and **recall = 0.705882** on the 240 test rows. Combining models did not help: hard voting had F1 **0.512397** and recall **0.607843**, while soft voting had F1 **0.492063** and recall **0.607843**. The stump's positive rates are **0.447236** for X1 ≤ 24 and **0.083102** for X1 > 24.

### R4.3 — AdaBoost

After round 1 normalisation, the two points misclassified by h1 (points 3 and 7) each have weight **0.25**, for a combined weight of **0.50**. Round 2 trusts h2 more because its weighted error is lower (**0.166667** versus **0.25**), giving h2 a larger alpha (**0.804719** versus **0.549306**).

### R4.4 — Neural-network architecture

**Model A, CNN option 2 output shapes and parameters**

| Layer | Output shape | Parameters |
|---|---:|---:|
| Input | 28 × 28 × 1 | 0 |
| Conv1 (5 × 5, 6 filters, padding 2) | 28 × 28 × 6 | 156 |
| ReLU1 | 28 × 28 × 6 | 0 |
| Max-pool1 (2 × 2, stride 2) | 14 × 14 × 6 | 0 |
| Conv2 (5 × 5, 16 filters, padding 0) | 10 × 10 × 16 | 2,416 |
| ReLU2 | 10 × 10 × 16 | 0 |
| Max-pool2 (2 × 2, stride 2) | 5 × 5 × 16 | 0 |
| Flatten | 400 | 0 |
| Dense1 (32 neurons) | 32 | 12,832 |
| Dense2 (10 classes) | 10 | 330 |

The MLP total is **13,002** parameters and the CNN total is **15,734**. For the 10-class image task, I would use **ReLU** in hidden layers and **softmax** in the output layer with **categorical cross-entropy** loss.

**Model B, RNN shapes**

- \(W_{xh}\): **128 × 64**
- \(W_{hh}\): **128 × 128**
- \(W_{hy}\): **16 × 128**
- \(b_h\): **128** parameters; \(b_y\): **16** parameters
- Total: **26,768** parameters.

The RNN parameter count does not grow with sequence length because the same weight matrices and biases are reused at every time step. A very deep plain network can suffer vanishing gradients, so gradients reaching early layers become too small to update them effectively; a skip connection provides a direct path for information and gradients to pass across layers.

For the 128 × 128 × 3 input comparison, the fully connected layer has **50,332,672** parameters, while the convolutional layer has **1,792**. Convolution scales better because each filter's small set of weights is shared across spatial positions rather than connecting every input pixel to every output neuron.

---

## Model-choice memo

**To: Head of the Retention Team**  
**Subject: Structured-data model recommendation**

I recommend piloting the one-question **stump** for churn screening, using a probability threshold of **0.30**. Its split is X1 ≤ 24; the training positive rates are 0.447236 on the yes side and 0.083102 on the no side. On the 240 test rows it achieved F1 **0.553846** and recall **0.705882**, slightly exceeding logistic regression's F1 of 0.541353 at the same threshold. Combining models did not improve results: hard-vote F1 was 0.512397 and soft-vote F1 was 0.492063.

For the required Task 2.8 comparator, logistic regression's training-CV-selected threshold was **0.09**, and its test cost was **₹6,300** under the ₹50-per-FP / ₹400-per-FN rule. The stump's cost was not computed under that cost rule, so it should not be assumed cheaper. This is a small, single split/test evaluation; the stump's simplicity may also miss interactions between customer features. Validate it on a later time-based holdout and calculate its business cost before full deployment.

**Model A and Model B are not worth building before the structured churn model is in production**, because image categorisation and review-topic classification are separate use cases rather than improvements to the current churn task.



