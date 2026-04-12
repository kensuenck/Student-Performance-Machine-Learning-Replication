import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the student mathematics dataset.
    The dataset is usually semicolon-separated.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {file_path}")

    df = pd.read_csv(path, sep=";")
    return df


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary pass/fail target from G3.
    pass = 1 if G3 >= 10 else 0
    """
    if "G3" not in df.columns:
        raise ValueError("Column 'G3' not found in dataset.")

    df = df.copy()
    df["pass"] = (df["G3"] >= 10).astype(int)
    return df


def run_experiment(df: pd.DataFrame, include_prior_grades: bool = True) -> None:
    """
    Train and evaluate Decision Tree and Random Forest models.
    """
    target = "pass"

    # Drop final grade from predictors to avoid leakage
    feature_cols = [c for c in df.columns if c not in ["G3", "pass"]]

    # Optional setup without prior grades
    if not include_prior_grades:
        feature_cols = [c for c in feature_cols if c not in ["G1", "G2"]]

    X = df[feature_cols]
    y = df[target]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical_cols,
            ),
        ]
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    setup_name = "WITH G1/G2" if include_prior_grades else "WITHOUT G1/G2"
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {setup_name}")
    print("=" * 70)

    for model_name, model in models.items():
        clf = Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"\n{model_name}")
        print("-" * len(model_name))
        print(f"Accuracy : {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall   : {rec:.3f}")
        print(f"F1-score : {f1:.3f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, digits=3, zero_division=0))


def main():
    # File is expected to be in the same folder as this script
    file_path = "student-mat.csv"

    df = load_data(file_path)
    df = prepare_target(df)

    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nTarget distribution:")
    print(df["pass"].value_counts(normalize=True).rename("proportion"))

    # Experiment 1: with prior grades
    run_experiment(df, include_prior_grades=True)

    # Experiment 2: without prior grades
    run_experiment(df, include_prior_grades=False)


if __name__ == "__main__":
    main()
