import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def main():

    X = np.array([
        [10, 1.2, 1.1, 3.0, 5.0, 4, 0, 0, 0, 0, 0],
        [2, 4.5, 3.9, 0.5, 3.2, 1, 1, 1, 1, 1, 1],
        [6, 2.1, 1.8, 1.2, 2.8, 2, 0, 1, 0, 1, 0],
        [1, 6.2, 5.5, 0.3, 1.1, 0, 1, 1, 1, 1, 1],
    ])

    # Labels: 0=LOW, 1=MEDIUM, 2=HIGH
    y = np.array([0, 2, 1, 2])

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X, y)

    os.makedirs("model_2", exist_ok=True)
    joblib.dump(model, "model_2/severity_model.pkl")

    print("[OK] severity_model.pkl created successfully")

if __name__ == "__main__":
    main()
