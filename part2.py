import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("records.csv")

# Select seven independent variables
features = [
    "customer_tenure_months",
    "complaint_count_90d",
    "discount_order_share",
    "avg_order_value_inr_hundreds",
    "loyalty_score",
    "cod_flag",
    "plus_member_flag"
]

X = df[features].to_numpy(dtype=float)
y = df["churned"].to_numpy(dtype=int)

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)
print("Positive cases:", np.sum(y == 1))
print("Negative cases:", np.sum(y == 0))

#Stratified 70/30 train-test split 70% training data = 560 records. 30% testing data = 240 records.
def stratified_split(X, y, test_size=0.30, seed=42):

    rng = np.random.default_rng(seed)

    train_indices = []
    test_indices = []

    for class_value in np.unique(y):

        indices = np.where(y == class_value)[0]

        rng.shuffle(indices)

        n_test = int(round(len(indices) * test_size))

        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices]
    )


X_train, X_test, y_train, y_test = stratified_split(X, y)

print("Training rows:", len(X_train))
print("Training positives:", np.sum(y_train == 1))

print("Testing rows:", len(X_test))
print("Testing positives:", np.sum(y_test == 1))

'''Standardise X1–X7

Standardisation is important because your features have different numerical ranges.

Formula: [ Z=\frac{X-\mu}{s} ]

Where:

(X) = original feature value
(\mu) = training mean
(s) = training sample standard deviation'''

mean = np.mean(X_train, axis=0)

std = np.std(X_train, axis=0, ddof=1)

X_train_scaled = (X_train - mean) / std

X_test_scaled = (X_test - mean) / std

print("Training means:", mean)
print("Training SD:", std)

print("Scaled training mean:",
      np.mean(X_train_scaled, axis=0))

print("Scaled training SD:",
      np.std(X_train_scaled, axis=0, ddof=1))


'''Implement sigmoid function Logistic Regression predicts a probability between 0 and '''1. The sigmoid formula is: [ \sigma(z)=\frac{1}{1+e^{-z}} ]'''
def sigmoid(z):

    z = np.clip(z, -500, 500)

    return 1 / (1 + np.exp(-z))

print(sigmoid(0))
print(sigmoid(2))
print(sigmoid(-2))

Implement Binary Cross-Entropy loss

'''Binary Cross-Entropy measures how far the model's predicted probabilities are from the actual answers. [ L=-\frac{1}{n}\sum_{i=1}^{n} [y_i\log(p_i)+(1-y_i)\log(1-p_i)] ]'''

def binary_cross_entropy(y_true, probabilities):

    eps = 1e-15

    probabilities = np.clip(
        probabilities, eps, 1 - eps
    )

    loss = -np.mean(
        y_true * np.log(probabilities)
        +
        (1 - y_true) * np.log(1 - probabilities)
    )

    return loss

initial_probabilities = np.full(len(y_train), 0.5)

initial_loss = binary_cross_entropy(
    y_train,
    initial_probabilities
)

print("Initial loss:", initial_loss)

'''This is because the model initially predicts 50% probability for every customer.

Implement Logistic Regression from scratch We will implement:

Model weights
Bias
Prediction probabilities
Gradient calculation
Gradient descent
1,000 updates
Loss tracking
The logistic regression model is:

[ p=\sigma(Xw+b) ] Gradient descent updates the parameters to minimise the loss.'''

def train_logistic_regression(
    X,
    y,
    learning_rate=0.1,
    iterations=1000
):

    n_samples, n_features = X.shape

    # Initialise weights and bias
    weights = np.zeros(n_features)
    bias = 0.0

    losses = []

    for iteration in range(iterations):

        # Forward propagation
        z = np.dot(X, weights) + bias

        probabilities = sigmoid(z)

        # Calculate loss
        loss = binary_cross_entropy(
            y, probabilities
        )

        losses.append(loss)

        # Calculate gradients
        dw = np.dot(
            X.T, (probabilities - y)
        ) / n_samples

        db = np.mean(probabilities - y)

        # Gradient descent
        weights -= learning_rate * dw

        bias -= learning_rate * db

    return weights, bias, losses

'''Train model first'''

weights, bias, losses = train_logistic_regression(
    X_train_scaled,
    y_train,
    learning_rate=0.1,
    iterations=1000
)

print("Final weights:", weights)
print("Final bias:", bias)
print("Final loss:", losses[-1])

plt.figure(figsize=(8, 5))

plt.plot(losses)

plt.xlabel("Iterations")
plt.ylabel("Binary Cross-Entropy Loss")

plt.title("Logistic Regression Training Loss")

plt.grid()

plt.savefig("part2_loss_curve.png")

plt.show()
'''Experiment with three learning rates assignment requires: 0.01 0.1 1.0 will compare their final losses.'''
learning_rates = [0.01, 0.1, 1.0]

lr_results = []

for lr in learning_rates:

    w, b, loss_history = train_logistic_regression(
        X_train_scaled,
        y_train,
        learning_rate=lr,
        iterations=1000
    )

    lr_results.append({
        "Learning Rate": lr,
        "Initial Loss": loss_history[0],
        "Final Loss": loss_history[-1]
    })

lr_df = pd.DataFrame(lr_results)

print(lr_df)

lr_df.to_csv(
    "part2_learning_rates.csv",
    index=False
)

lr_df.to_csv(
    "part2_learning_rates.csv",
    index=False
)

'''Learning rate What to observe 0.01 Slower learning 0.1 Balanced learning 1.0 Faster updates; check stability

Do not automatically assume 0.1 is best. Compare the actual final losses.

Calculate odds ratios An odds ratio tells us how a one-unit increase in a standardised feature changes the odds of churn, holding the other features constant. [ OR=e^{\beta} ]'''

odds_ratios = np.exp(weights)

odds_df = pd.DataFrame({
    "Feature": features,
    "Weight": weights,
    "Odds_Ratio": odds_ratios
})

print(odds_df)

'''Interpretation:

OR greater than 1: higher odds of churn.
OR less than 1: lower odds of churn.
OR equal to 1: no change in odds.
these odds ratios refer to a one-standard-deviation increase in each feature because the inputs were standardised.

Generate predictions Now use the trained model to predict churn probabilities on test data.'''

test_probabilities = sigmoid(
    np.dot(X_test_scaled, weights) + bias
)

print(test_probabilities[:10])

'''We will test two thresholds:

0.5
0.3 A probability above or equal to the threshold is classified as churn.
Calculate TP, FP, TN and FN These four values are essential. Term Meaning TP Correctly predicted churn FP Predicted churn, but customer did not churn TN Correctly predicted non-churn FN Predicted non-churn, but customer churned'''

def confusion_counts(y_true, y_pred):

    TP = np.sum((y_true == 1) & (y_pred == 1))

    FP = np.sum((y_true == 0) & (y_pred == 1))

    TN = np.sum((y_true == 0) & (y_pred == 0))

    FN = np.sum((y_true == 1) & (y_pred == 0))

    return TP, FP, TN, FN

'''Calculate all evaluation metrics [ Accuracy=\frac{TP+TN}{TP+FP+TN+FN} ]

[ Precision=\frac{TP}{TP+FP} ]

[ Recall=\frac{TP}{TP+FN} ]

[ F1=\frac{2PR}{P+R} ]

[ Specificity=\frac{TN}{TN+FP} ] '''   

def calculate_metrics(y_true, y_pred):

    TP, FP, TN, FN = confusion_counts(
        y_true, y_pred
    )

    accuracy = (TP + TN) / len(y_true)

    precision = TP / (TP + FP) if TP + FP else 0

    recall = TP / (TP + FN) if TP + FN else 0

    specificity = TN / (TN + FP) if TN + FP else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity
    }

'''Test both threshold '''
threshold_results = []

for threshold in [0.5, 0.3]:

    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        y_test, predictions
    )

    metrics["Threshold"] = threshold

    threshold_results.append(metrics)

threshold_df = pd.DataFrame(
    threshold_results
)

print(threshold_df)

threshold_df.to_csv(
    "part2_thresholds.csv",
    index=False
)

'''Business interpretation: Lowering the threshold from 0.5 to 0.3 generally identifies more potential churners. It can increase recall but may also increase false positives.

Implement KNN from scratch Now we move to the second classification model: K-Nearest Neighbours. KNN works by finding the nearest customers based on their feature values and predicting the majority class among those neighbours. Distance formula (Euclidean distance): [ d(x,y)=\sqrt{\sum_{j=1}^{7}(x_j-y_j)^2} ]'''

def knn_predict(X_train, y_train, X_test, k):

    predictions = []

    for test_point in X_test:

        # Calculate distances
        distances = np.sqrt(
            np.sum((X_train - test_point) ** 2, axis=1)
        )

        # Find nearest k neighbours
        nearest_indices = np.argsort(distances)[:k]

        # Get their labels
        nearest_labels = y_train[nearest_indices]

        # Majority voting
        prediction = int(
            np.sum(nearest_labels) > k / 2
        )

        predictions.append(prediction)

    return np.array(predictions)

knn_predictions = knn_predict(
    X_train_scaled,
    y_train,
    X_test_scaled,
    k=5
)

print(knn_predictions[:20])   

'''Implement 5-fold cross-validation assignment requires testing: [ K=1,3,5,7,9,11,13,15,17,19,21 ] We will divide training data into five folds. Four folds for training. One fold for validation. Repeat five times. Calculate F1 for each fold. Calculate the average F1. We must not use the test dataset to choose K.'''
def make_stratified_folds(y, n_folds=5, seed=42):

    rng = np.random.default_rng(seed)

    folds = [[] for _ in range(n_folds)]

    for class_value in np.unique(y):

        indices = np.where(y == class_value)[0]

        rng.shuffle(indices)

        for i, index in enumerate(indices):

            folds[i % n_folds].append(index)

    return [
        np.array(fold, dtype=int)
        for fold in folds
    ]
'''Now implement cross-validation:'''

def cross_validate_knn(X, y, k_values):

    folds = make_stratified_folds(
        y, n_folds=5
    )

    results = []

    for k in k_values:

        fold_f1_scores = []

        for i in range(5):

            validation_indices = folds[i]

            training_indices = np.concatenate([
                folds[j]
                for j in range(5)
                if j != i
            ])

            X_fold_train = X[training_indices]
            y_fold_train = y[training_indices]

            X_fold_val = X[validation_indices]
            y_fold_val = y[validation_indices]

            predictions = knn_predict(
                X_fold_train,
                y_fold_train,
                X_fold_val,
                k
            )

            metrics = calculate_metrics(
                y_fold_val,
                predictions
            )

            fold_f1_scores.append(
                metrics["F1"]
            )

        results.append({
            "K": k,
            "Mean_F1": np.mean(fold_f1_scores),
            "Std_F1": np.std(fold_f1_scores)
        })

    return pd.DataFrame(results)

k_values = list(range(1, 22, 2))

knn_results = cross_validate_knn(
    X_train_scaled,
    y_train,
    k_values
)

print(knn_results)    

'''Find the best K'''

best_k = int(
    knn_results.loc[
        knn_results["Mean_F1"].idxmax(),
        "K"
    ]
)

print("Best K:", best_k)

print("Best Mean F1:",
      knn_results["Mean_F1"].max())

'''Plot the result'''
plt.figure(figsize=(8, 5))

plt.plot(
    knn_results["K"],
    knn_results["Mean_F1"],
    marker="o"
)

plt.xlabel("K Value")
plt.ylabel("Mean Cross-Validation F1")

plt.title("KNN: 5-Fold Cross Validation")

plt.grid()

plt.savefig("part2_knn_cv.png")

plt.show()

'''Calculate financial cost [ Cost=50(FP)+400(FN) ] Every false positive costs ₹50. Every false negative costs ₹400. A false negative is much more expensive because we fail to identify a customer who actually churns.'''

for threshold in [0.5, 0.3]:

    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    TP, FP, TN, FN = confusion_counts(
        y_test, predictions
    )

    cost = financial_cost(FP, FN)

    print("\nThreshold:", threshold)

    print("False Positives:", FP)

    print("False Negatives:", FN)

    print("Financial Cost: ₹", cost)

'''Find the best threshold using training data only Do not use test labels to select your threshold. We will generate out-of-fold predictions on training data and use these to compare thresholds based on financial cost.'''
def get_oof_logistic_probabilities(
    X,
    y,
    learning_rate=0.1,
    iterations=1000
):

    folds = make_stratified_folds(
        y, n_folds=5
    )

    oof_probabilities = np.zeros(len(y))

    for i in range(5):

        val_idx = folds[i]

        train_idx = np.concatenate([
            folds[j]
            for j in range(5)
            if j != i
        ])

        w, b, _ = train_logistic_regression(
            X[train_idx],
            y[train_idx],
            learning_rate,
            iterations
        )

        oof_probabilities[val_idx] = sigmoid(
            np.dot(X[val_idx], w) + b
        )

    return oof_probabilities

oof_probabilities = get_oof_logistic_probabilities(
    X_train_scaled,
    y_train
)

threshold_candidates = np.arange(
    0.05, 0.96, 0.01
)

cost_results = []

for threshold in threshold_candidates:

    predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    TP, FP, TN, FN = confusion_counts(
        y_train, predictions
    )

    cost = financial_cost(FP, FN)

    cost_results.append({
        "Threshold": threshold,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "Cost": cost
    })

cost_df = pd.DataFrame(cost_results)

best_row = cost_df.loc[
    cost_df["Cost"].idxmin()
]

best_threshold = best_row["Threshold"]

print("Best threshold:", best_threshold)

print("Minimum training CV cost:", best_row["Cost"])

'''We have now selected a threshold without looking at test outcomes.

Evaluate the final logistic regression model at this threshold:'''

final_predictions = (
    test_probabilities >= best_threshold
).astype(int)

final_metrics = calculate_metrics(
    y_test,
    final_predictions
)

final_cost = financial_cost(
    final_metrics["FP"],
    final_metrics["FN"]
)

print("Final test metrics:", final_metrics)

print("Final test financial cost:", final_cost)

'''Evaluate the best KNN model Now use your selected K on the test set.'''

best_knn_predictions = knn_predict(
    X_train_scaled,
    y_train,
    X_test_scaled,
    best_k
)

best_knn_metrics = calculate_metrics(
    y_test,
    best_knn_predictions
)

print("Best K:", best_k)

print("KNN test metrics:")

for metric, value in best_knn_metrics.items():

    print(metric, ":", value)


'''For the financial cost:'''

knn_cost = financial_cost(
    best_knn_metrics["FP"],
    best_knn_metrics["FN"]
)

print("KNN financial cost: ₹", knn_cost)

# 1. Save logistic regression losses

pd.DataFrame({
    "Iteration": np.arange(1, len(losses) + 1),
    "Loss": losses
}).to_csv(
    "part2_losses.csv",
    index=False
)


# 2. Save weights and odds ratios

weights_df = pd.DataFrame({
    "Feature": features,
    "Weight": weights,
    "Odds_Ratio": np.exp(weights)
})

weights_df.to_csv(
    "part2_weights.csv",
    index=False
)

# 3. Save threshold evaluation and financial costs

final_threshold_rows = []

for threshold in [0.5, 0.3, best_threshold]:

    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        y_test, predictions
    )

    metrics["Threshold"] = threshold

    metrics["Financial_Cost"] = financial_cost(
        metrics["FP"],
        metrics["FN"]
    )

    metrics["Model"] = "Logistic Regression"

    final_threshold_rows.append(metrics)


knn_row = best_knn_metrics.copy()

knn_row["Threshold"] = np.nan

knn_row["Financial_Cost"] = knn_cost

knn_row["Model"] = f"KNN (K={best_k})"

final_threshold_rows.append(knn_row)

pd.DataFrame(
    final_threshold_rows
).to_csv(
    "part2_thresholds.csv",
    index=False
)

print("All three Part 2 files saved successfully!")

'''save the KNN cross-validation table separately'''

knn_results.to_csv(
    "part2_knn_cv.csv",
    index=False
)



    



