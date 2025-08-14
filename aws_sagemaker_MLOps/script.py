import os
import argparse
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--train-file", type=str, default="train-v1.csv")
    parser.add_argument("--test-file", type=str, default="test-v1.csv")
    args, _ = parser.parse_known_args()

    # Load data
    train_data = pd.read_csv(os.path.join(args.train, args.train_file))
    test_data = pd.read_csv(os.path.join(args.test, args.test_file))

    # Features & labels
    features = list(train_data.columns)
    label = features.pop(-1)
    X_train, y_train = train_data[features], train_data[label]
    X_test, y_test = test_data[features], test_data[label]

    # Train model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        verbose=2
    )
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

    # Evaluate
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))
