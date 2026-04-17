import pandas as pd
from pathlib import Path

# Import train/test split function
from sklearn.model_selection import train_test_split

# Tools for preprocessing different column types
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# Machine learning models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    The dataset uses semicolon (;) as a separator.
    """
    path = Path(file_path)

    # Check if file exists before loading
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {file_path}")

    # Read CSV file
    df = pd.read_csv(path, sep=";")
    return df


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target variable 'pass' based on final grade (G3).
    pass = 1 if G3 >= 10 (student passes)
    pass = 0 if G3 < 10 (student fails)
    """
    # Ensure target column exists
    if "G3" not in df.columns:
        raise ValueError("Column 'G3' not found in dataset.")

    df = df.copy()

    # Create binary classification target
    df["pass"] = (df["G3"] >= 10).astype(int)
    return df


def run_experiment(df: pd.DataFrame, include_prior_grades: bool = True) -> None:
    """
    Train and evaluate Decision Tree and Random Forest models.
    Two setups:
    - WITH prior grades (G1, G2)
    - WITHOUT prior grades (fairness / early prediction scenario)
    """

    target = "pass"

    # Select feature columns (exclude final grade and target to avoid leakage)
    feature_cols = [c for c in df.columns if c not in ["G3", "pass"]]

    # Optionally remove prior grades to test fairness / early prediction
    if not include_prior_grades:
        feature_cols = [c for c in feature_cols if c not in ["G1", "G2"]]

    # Split features and target
    X = df[feature_cols]
    y = df[target]

    # Automatically detect numeric and categorical columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Preprocessing pipeline:
    # - Numeric: fill missing values with median
    # - Categorical: fill missing values + one-hot encode
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

    # Define models
    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,        # limit depth to reduce overfitting
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,  # number of trees
            max_depth=8,       # control complexity
            random_state=42
        ),
    }

    # Split data into training and test sets
    # Stratify ensures same class balance in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Label experiment setup
    setup_name = "WITH G1/G2" if include_prior_grades else "WITHOUT G1/G2"
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {setup_name}")
    print("=" * 70)

    # Train and evaluate each model
    for model_name, model in models.items():

        # Combine preprocessing and model into one pipeline
        clf = Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])

        # Train model
        clf.fit(X_train, y_train)

        # Generate predictions on test data
        y_pred = clf.predict(X_test)

        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Print results
        print(f"\n{model_name}")
        print("-" * len(model_name))
        print(f"Accuracy : {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall   : {rec:.3f}")
        print(f"F1-score : {f1:.3f}")

        # Detailed classification report
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, digits=3, zero_division=0))


def main():
    """
    Main execution function.
    Runs the full replication in one execution (assignment requirement).
    """

    # Dataset should be in same directory
    file_path = "student-mat.csv"

    # Load and prepare data
    df = load_data(file_path)
    df = prepare_target(df)

    # Print basic dataset info
    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Show class distribution (important for evaluation)
    print("\nTarget distribution:")
    print(df["pass"].value_counts(normalize=True).rename("proportion"))

    # Run both experiments
    # 1. With prior grades (high accuracy scenario)
    run_experiment(df, include_prior_grades=True)

    # 2. Without prior grades (fairness / early prediction scenario)
    run_experiment(df, include_prior_grades=False)


# Ensure script runs only when executed directly
if __name__ == "__main__":
    main()
