import numpy as np
import os

from src.IQL.learning.class_state import ClassState
from src.IQL.encoding.embedding_to_state import embedding_to_state
from src.IQL.regimes.regime2_online import OnlinePerceptron
from src.IQL.learning.metrics import summarize_training
from src.IQL.backends.exact import ExactBackend
from src.utils.paths import load_paths
from src.utils.seed import set_seed
from src.utils.load_data import load_data

# ----------------------------
# Reproducibility
# ----------------------------
set_seed(42)

def main():
    X_train, X_test, y_train, y_test = load_data("polar")

    chi0 = np.zeros_like(X_train[0])
    for psi, label in zip(X_train[:10], y_train[:10]):
        chi0 += label * psi
    chi0 = chi0 / np.linalg.norm(chi0)

    class_state = ClassState(chi0,backend=ExactBackend(),label=+1)
    trainer = OnlinePerceptron(class_state, eta=0.1)

    acc = trainer.fit(X_train,y_train)
    stats = summarize_training(trainer.history)

    print("Final accuracy:", acc)
    print("Training stats:", stats)


if __name__ == "__main__":
    main()

### output 
"""
🌱 Global seed set to 42
Loaded train embeddings: (3500, 32)
Final accuracy: 0.8562857142857143
Training stats: {'mean_margin': 0.14930659062683652, 'min_margin': -0.7069261085786833, 'num_updates': 503, 'update_rate': 0.1437142857142857}
"""