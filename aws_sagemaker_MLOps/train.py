# train.py
import os
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)

    # SageMaker-provided paths (with local fallbacks for dev/test)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "./"))
    parser.add_argument("--test",  type=str, default=os.environ.get("SM_CHANNEL_TEST", "./"))

    # Filenames within those dirs
    parser.add_argument("--train-file", type=str, default="train-v1.csv")
    parser.add_argument("--test-file",  type=str, default="test-v1.csv")
    args, _ = parser.parse_known_args()

    print("Args:", args)

    # Ensure dirs exist locally
    os.makedirs(args.model_dir, exist_ok=True)

    # Load data
    train_path = os.path.join(args.train, args.train_file)
    test_path  = os.path.join(args.test,  args.test_file)

    print("Reading:", train_path)
    print("Reading:", test_path)

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Features & label
    cols = list(train_df.columns)
    label = cols[-1]   # last column is label (train-v1.csv layout)
    feature_cols = cols[:-1]

    X_train, y_train = train_df[feature_cols], train_df[label]
    X_test,  y_test  = test_df[feature_cols],  test_df[label]

    print("Training shapes:", X_train.shape, y_train.shape)
    print("Testing shapes :", X_test.shape,  y_test.shape)

    # Train
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
        verbose=1
    )
    clf.fit(X_train, y_train)

    # Save model to model-dir (SageMaker will capture this)
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(clf, model_path)
    print(f"Model saved to: {model_path}")

    # Quick eval
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, preds))

if __name__ == "__main__":
    main()
