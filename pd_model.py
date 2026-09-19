"""
Task 3 & 4: Probability of Default (PD) model + Expected Loss function
------------------------------------------------------------------------
Trains several classifiers to estimate PD from borrower characteristics,
compares them, then wraps the best one in an `expected_loss()` function:

    Expected Loss = PD * EAD * (1 - Recovery Rate)

EAD (exposure at default) = loan_amt_outstanding
Recovery rate = 10% (given) -> LGD = 90%
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

RECOVERY_RATE = 0.10
_HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv(_HERE / "Task_3_and_4_Loan_Data.csv")

FEATURES = ["credit_lines_outstanding", "loan_amt_outstanding", "total_debt_outstanding",
            "income", "years_employed", "fico_score"]
X = df[FEATURES]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Logistic regression benefits from scaled features; tree models don't need it
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 2. Train multiple candidate models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), True),   # (model, needs_scaling)
    "Decision Tree": (DecisionTreeClassifier(max_depth=5, random_state=42), False),
    "Random Forest": (RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42), False),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42), False),
}

results = {}
for name, (model, needs_scaling) in models.items():
    Xtr, Xte = (X_train_scaled, X_test_scaled) if needs_scaling else (X_train, X_test)
    model.fit(Xtr, y_train)
    proba = model.predict_proba(Xte)[:, 1]
    preds = (proba >= 0.5).astype(int)
    results[name] = {
        "model": model,
        "needs_scaling": needs_scaling,
        "auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "proba": proba,
    }

# ---------------------------------------------------------------
# 3. Comparative analysis
# ---------------------------------------------------------------
def print_comparison():
    print(f"{'Model':<22}{'AUC':>8}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
    for name, r in results.items():
        print(f"{name:<22}{r['auc']:>8.4f}{r['accuracy']:>10.4f}{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1']:>8.4f}")

def plot_roc_curves(save_path=_HERE / "pd_model_comparison_roc.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: PD Model Comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# Pick the best model by AUC (the standard metric for a PD ranking model)
BEST_MODEL_NAME = max(results, key=lambda n: results[n]["auc"])

# ---------------------------------------------------------------
# 4. Public functions: predict_default_probability() and expected_loss()
# ---------------------------------------------------------------
def predict_default_probability(credit_lines_outstanding, loan_amt_outstanding,
                                 total_debt_outstanding, income, years_employed, fico_score,
                                 model_name=None):
    """Return PD (0-1) for a borrower, using the given model (defaults to the best-AUC model)."""
    name = model_name or BEST_MODEL_NAME
    if name not in results:
        raise ValueError(f"Unknown model_name '{name}'. Choose from: {list(results.keys())}")
    model = results[name]["model"]
    needs_scaling = results[name]["needs_scaling"]

    row = pd.DataFrame([{
        "credit_lines_outstanding": credit_lines_outstanding,
        "loan_amt_outstanding": loan_amt_outstanding,
        "total_debt_outstanding": total_debt_outstanding,
        "income": income,
        "years_employed": years_employed,
        "fico_score": fico_score,
    }])[FEATURES]

    X_input = scaler.transform(row) if needs_scaling else row
    return float(model.predict_proba(X_input)[0, 1])


def expected_loss(credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding,
                   income, years_employed, fico_score, recovery_rate=RECOVERY_RATE, model_name=None):
    """
    Expected Loss = PD * EAD * (1 - recovery_rate)
    EAD (exposure at default) is taken as loan_amt_outstanding.
    """
    pd_estimate = predict_default_probability(
        credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding,
        income, years_employed, fico_score, model_name=model_name,
    )
    ead = loan_amt_outstanding
    lgd = 1 - recovery_rate
    return round(pd_estimate * ead * lgd, 2)


if __name__ == "__main__":
    # self-check: expected_loss must equal PD * EAD * (1 - recovery_rate), always
    _s = dict(credit_lines_outstanding=3, loan_amt_outstanding=7000, total_debt_outstanding=6000,
              income=40000, years_employed=2, fico_score=610)
    _pd = predict_default_probability(**_s)
    assert expected_loss(**_s) == round(_pd * _s["loan_amt_outstanding"] * 0.9, 2)

    print("=" * 70)
    print("MODEL COMPARISON (test set, 25% holdout)")
    print("=" * 70)
    print_comparison()
    print(f"\nBest model by AUC: {BEST_MODEL_NAME}\n")

    plot_roc_curves()
    print("Saved ROC comparison plot.\n")

    # Feature importance / coefficients for interpretability
    print("=" * 70)
    print("Logistic Regression coefficients (standardized features)")
    print("=" * 70)
    lr = results["Logistic Regression"]["model"]
    for feat, coef in zip(FEATURES, lr.coef_[0]):
        print(f"  {feat:<28} {coef:+.4f}")

    print()
    print("=" * 70)
    print("Random Forest feature importances")
    print("=" * 70)
    rf = results["Random Forest"]["model"]
    for feat, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:<28} {imp:.4f}")

    print()
    print("=" * 70)
    print(f"SAMPLE PREDICTIONS (using best model: {BEST_MODEL_NAME})")
    print("=" * 70)
    samples = [
        dict(credit_lines_outstanding=0, loan_amt_outstanding=5000, total_debt_outstanding=4000,
             income=75000, years_employed=5, fico_score=700),   # low risk profile
        dict(credit_lines_outstanding=4, loan_amt_outstanding=8000, total_debt_outstanding=9000,
             income=27000, years_employed=1, fico_score=580),   # high risk profile
        dict(credit_lines_outstanding=2, loan_amt_outstanding=6000, total_debt_outstanding=5500,
             income=45000, years_employed=3, fico_score=630),   # mid risk profile
    ]
    for s in samples:
        pd_est = predict_default_probability(**s)
        el = expected_loss(**s)
        print(f"PD={pd_est:.3f}  loan_amt=${s['loan_amt_outstanding']:,.0f}  "
              f"fico={s['fico_score']}  -> Expected Loss = ${el:,.2f}")
