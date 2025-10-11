import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

scoring = {
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
}


def summarize_metric(model_name, results, type="cv"):
    performance = {"model_name": model_name}

    for metric in scoring:
        if type == "cv":
            train_scores = results[f"train_{metric}"]
            test_scores = results[f"test_{metric}"]

            if "neg_" in scoring[metric]:
                train_scores = -train_scores
                test_scores = -test_scores

            print(f"{metric.upper()} on train set: {train_scores.mean():.3f}")
            print(f"{metric.upper()} on validation set: {test_scores.mean():.3f}")

            performance[f"train_{metric.upper()}"] = train_scores.mean()
            performance[f"val_{metric.upper()}"] = test_scores.mean()
        else:
            train_score = results.cv_results_[f"mean_train_{metric}"][
                results.best_index_
            ]
            test_score = results.cv_results_[f"mean_test_{metric}"][results.best_index_]

            if "neg_" in scoring[metric]:
                train_score = -train_score
                test_score = -test_score

            print(f"Best {metric.upper()} on train set: {train_score:.3f}")
            print(f"Best {metric.upper()} on validation set: {test_score:.3f}")

            performance[f"train_{metric.upper()}"] = train_score
            performance[f"val_{metric.upper()}"] = test_score

    return performance


def plot_predicted_vs_actual(y_train, y_pred, model_name):
    min_val = min(y_train.min(), y_pred.min())
    max_val = max(y_train.max(), y_pred.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(y_train, y_pred, alpha=0.5, s=10)
    plt.plot([min_val, max_val], [min_val, max_val], "r--")  # 45° line
    plt.title(f"{model_name}: Predicted vs Actual", fontsize=16)
    plt.xlabel("Actual Values", fontsize=14)
    plt.ylabel("Predicted Values", fontsize=14)
    plt.tight_layout()
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_residuals(y_train, y_pred, model_name):
    residuals = y_train - y_pred

    plt.figure(figsize=(6, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, s=10)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(f"{model_name}: Residuals vs Predicted Values", fontsize=16)
    plt.xlabel("Predicted Values", fontsize=14)
    plt.ylabel("Residuals (True – Predicted)", fontsize=14)
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_error_contribution(y_train, y_pred):
    errors = y_train - y_pred
    squared_errors = errors**2

    df_errors = pd.DataFrame(
        {
            "y_true": y_train,
            "y_pred": y_pred,
            "error": errors,
            "sq_error": squared_errors,
        }
    )

    df_errors["rank"] = np.arange(
        len(df_errors)
    )  # to keep track of original order if needed

    top10 = df_errors.sort_values("sq_error", ascending=False).head(10)
    print("Top 10 samples by MSE contribution:")
    print(top10)

    plt.figure(figsize=(8, 4))
    plt.bar(
        np.arange(len(df_errors)),
        df_errors["sq_error"].sort_values().values,
        edgecolor="k",
        linewidth=0.2,
    )
    plt.xlabel("Sample index (sorted by contribution)")
    plt.ylabel("Squared error contribution")
    plt.title("Per-sample contributions to MSE loss")
    plt.tight_layout()
    plt.show()
