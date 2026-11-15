"""Part 4 solution. Run from this folder with: python part4.py"""
from pathlib import Path
import numpy as np
import pandas as pd
from model_helpers import (
    FEATURES, stratified_split, sigmoid, train_logistic_regression,
    knn_probabilities, cross_validate_knn, calculate_metrics,
    confusion_counts, get_oof_logistic_probabilities, financial_cost,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)
DATA = ROOT / "records.csv"

def entropy(labels):
    if len(labels) == 0:
        return 0.0
    p = np.mean(labels)
    terms = [q * np.log2(q) for q in (p, 1-p) if q > 0]
    return float(-sum(terms))

def gini(labels):
    if len(labels) == 0:
        return 0.0
    p = np.mean(labels)
    return float(1 - p**2 - (1-p)**2)

def score_splits(X_train, y_train):
    # X1...X7 follow the feature order in model_helpers.FEATURES.
    conditions = {
        "S1": X_train[:, 5] == 1,
        "S2": X_train[:, 6] == 1,
        "S3": X_train[:, 2] > 0.5,
        "S4": X_train[:, 0] <= 24,
    }
    root_h, root_g = entropy(y_train), gini(y_train)
    rows = []
    for name, yes_mask in conditions.items():
        y_yes, y_no = y_train[yes_mask], y_train[~yes_mask]
        weighted_h = (len(y_yes)*entropy(y_yes) + len(y_no)*entropy(y_no))/len(y_train)
        weighted_g = (len(y_yes)*gini(y_yes) + len(y_no)*gini(y_no))/len(y_train)
        rows.append({
            "split": name, "n_yes": len(y_yes), "positives_yes": int(y_yes.sum()),
            "n_no": len(y_no), "positives_no": int(y_no.sum()),
            "info_gain": root_h-weighted_h, "gini_gain": root_g-weighted_g,
        })
    return root_h, root_g, pd.DataFrame(rows), conditions

def run_adaboost():
    y = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=int)
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1], dtype=int)
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1], dtype=int)
    weights = np.full(8, 1/8, dtype=float)
    rows, errors, alphas = [], [], []
    for round_no, h in ((1, h1), (2, h2)):
        before = weights.copy()
        error = float(before[h != y].sum())
        alpha = float(0.5 * np.log((1-error)/error))
        unnormalised = before * np.exp(-alpha * y * h)
        after = unnormalised / unnormalised.sum()
        errors.append(error)
        alphas.append(alpha)
        for i in range(8):
            rows.append({
                "round": round_no, "point": i+1,
                "weight_before": before[i],
                "weight_unnormalised": unnormalised[i],
                "weight_after": after[i],
            })
        weights = after
    ensemble = np.sign(alphas[0]*h1 + alphas[1]*h2)
    accuracy = float(np.mean(ensemble == y))
    return errors, alphas, pd.DataFrame(rows), accuracy

def architecture():
    # MLP: 784 -> 16 -> 16 -> 10
    mlp = [
        ("mlp_hidden1", 784*16 + 16),
        ("mlp_hidden2", 16*16 + 16),
        ("mlp_output", 16*10 + 10),
    ]
    # CNN spatial sizes: 28 -> conv1 28 -> pool 14 -> conv2 10 -> pool 5.
    cnn = [
        ("cnn_conv1", (5*5*1 + 1)*6),
        ("cnn_conv2", (5*5*6 + 1)*16),
        ("cnn_dense1", (5*5*16)*32 + 32),
        ("cnn_dense2", 32*10 + 10),
    ]
    # RNN parameter matrices and biases.
    rnn_total = 64*128 + 128*128 + 128*16 + 128 + 16
    fc_params = (128*128*3)*1024 + 1024
    conv_params = (3*3*3 + 1)*64
    rows = mlp + cnn
    return rows, sum(v for _, v in mlp), sum(v for _, v in cnn), fc_params, conv_params, rnn_total

def main():
    df = pd.read_csv(DATA)
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["churned"].to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = stratified_split(X, y, 0.30, 42)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=1)
    X_train_scaled = (X_train-mean)/std
    X_test_scaled = (X_test-mean)/std

    # 4.1: splits use original, unstandardised training units.
    root_h, root_g, split_df, conditions = score_splits(X_train, y_train)
    split_df.to_csv(OUTPUT/"part4_splits.csv", index=False)
    best_info = split_df.loc[split_df.info_gain.idxmax(), "split"]
    best_gini = split_df.loc[split_df.gini_gain.idxmax(), "split"]

    # 4.2: fit the Part 2 logistic model and select K using training-only CV.
    weights, bias, _ = train_logistic_regression(X_train_scaled, y_train, 0.1, 1000)
    p_log = sigmoid(X_test_scaled @ weights + bias)
    cv_rows = cross_validate_knn(X_train_scaled, y_train, range(1, 22, 2))
    best_k = max(cv_rows, key=lambda r: r["Mean_F1"])["K"]
    p_knn = knn_probabilities(X_train_scaled, y_train, X_test_scaled, best_k)
    pred_log = (p_log >= 0.3).astype(int)
    pred_knn = (p_knn >= ((best_k+1)/2)/best_k).astype(int)

    best_condition = conditions[best_info]
    stump_yes = float(y_train[best_condition].mean())
    stump_no = float(y_train[~best_condition].mean())
    p_stump = np.where(
        X_test[:, {"S1":5,"S2":6,"S3":2,"S4":0}[best_info]] == 1,
        stump_yes, stump_no
    ) if best_info in ("S1","S2") else (
        np.where(X_test[:,2] > 0.5, stump_yes, stump_no) if best_info=="S3"
        else np.where(X_test[:,0] <= 24, stump_yes, stump_no)
    )
    pred_stump = (p_stump >= 0.3).astype(int)
    pred_hard = ((pred_log + pred_knn + pred_stump) >= 2).astype(int)
    p_soft = (p_log + p_knn + p_stump)/3
    pred_soft = (p_soft >= 0.3).astype(int)

    voting_rows = []
    for name, pred in (
        ("logistic", pred_log), ("knn", pred_knn), ("stump", pred_stump),
        ("hard_vote", pred_hard), ("soft_vote", pred_soft),
    ):
        m = calculate_metrics(y_test, pred)
        voting_rows.append({
            "model": name, "tp": m["TP"], "fp": m["FP"], "tn": m["TN"], "fn": m["FN"],
            "precision": m["Precision"], "recall": m["Recall"], "f1": m["F1"],
        })
    pd.DataFrame(voting_rows).to_csv(OUTPUT/"part4_voting.csv", index=False)

    # Task 2.8 comparator: choose threshold from training-only OOF probabilities.
    oof = get_oof_logistic_probabilities(X_train_scaled, y_train)
    threshold_costs = []
    for threshold in np.arange(0.05, 0.96, 0.01):
        pred = (oof >= threshold).astype(int)
        tp, fp, tn, fn = confusion_counts(y_train, pred)
        threshold_costs.append((financial_cost(fp, fn), float(threshold)))
    logistic_cost_threshold = min(threshold_costs)[1]
    logistic_task28_test_pred = (p_log >= logistic_cost_threshold).astype(int)
    _, log_fp, _, log_fn = confusion_counts(y_test, logistic_task28_test_pred)
    logistic_task28_test_cost = financial_cost(log_fp, log_fn)

    # 4.3 AdaBoost.
    errors, alphas, ada_df, ensemble_accuracy = run_adaboost()
    ada_df.to_csv(OUTPUT/"part4_adaboost.csv", index=False)

    # 4.4 OOB share.
    oob_fraction = float((1 - 1/560)**560)

    # 4.5 parameter counts.
    arch_rows, mlp_total, cnn_total, fc_params, conv_params, rnn_total = architecture()
    pd.DataFrame(arch_rows, columns=["layer","parameters"]).to_csv(
        OUTPUT/"part4_architecture.csv", index=False
    )

    # Required result keys, one key per line.
    results = {
        "root_entropy": root_h,
        "root_gini": root_g,
        "best_split_by_info_gain": best_info,
        "best_split_by_gini_gain": best_gini,
        "stump_rate_yes": stump_yes,
        "stump_rate_no": stump_no,
        "best_k": best_k,
        "round1_error": errors[0],
        "round1_alpha": alphas[0],
        "round2_error": errors[1],
        "round2_alpha": alphas[1],
        "ensemble_accuracy": ensemble_accuracy,
        "oob_fraction": oob_fraction,
        "mlp_total": mlp_total,
        "cnn_total": cnn_total,
        "fc_layer_params": fc_params,
        "conv_layer_params": conv_params,
        "rnn_total": rnn_total,
        "logistic_task2_8_threshold": logistic_cost_threshold,
        "logistic_task2_8_test_cost_inr": logistic_task28_test_cost,
    }
    with (OUTPUT/"part4_results.txt").open("w", encoding="utf-8") as f:
        for key, value in results.items():
            f.write(f"{key} = {value:.10f}\n" if isinstance(value, (float, np.floating))
                    else f"{key} = {value}\n")

    print("Part 4 completed. Outputs saved in:", OUTPUT)
    print((OUTPUT/"part4_results.txt").read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
