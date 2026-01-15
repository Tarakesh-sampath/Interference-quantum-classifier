import os
import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from src.utils.paths import load_paths
from src.utils.seed import set_seed
set_seed(42)

# ----------------------------
# Load paths
# ----------------------------
BASE_ROOT, PATHS = load_paths()

EMBED_DIR = PATHS["embeddings"]
LOG_DIR = PATHS["logs"]
os.makedirs(LOG_DIR, exist_ok=True)

# ----------------------------
# Load embeddings
# ----------------------------
X = np.load(os.path.join(EMBED_DIR, "val_embeddings.npy"))
y = np.load(os.path.join(EMBED_DIR, "val_labels.npy"))

print("Loaded embeddings:", X.shape)

# ----------------------------
# Preprocessing
# ----------------------------
# 1) Standardize (important for linear models)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 2) L2-normalize (important for similarity & quantum)
X_l2 = normalize(X_std, norm="l2")

# ----------------------------
# Train / test split
# ----------------------------
Xtr_s, Xte_s, ytr, yte = train_test_split(
    X_std, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

Xtr_l2, Xte_l2, _, _ = train_test_split(
    X_l2, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

results = {}

# ==================================================
# 1️⃣ Logistic Regression (Linear separability)
# ==================================================
print("\nTraining Logistic Regression...")
logreg = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)
logreg.fit(Xtr_s, ytr)

pred_lr = logreg.predict(Xte_s)
proba_lr = logreg.predict_proba(Xte_s)[:, 1]

results["LogisticRegression"] = {
    "accuracy": accuracy_score(yte, pred_lr),
    "auc": roc_auc_score(yte, proba_lr)
}

# ==================================================
# 2️⃣ Linear SVM (Max-margin)
# ==================================================
print("Training Linear SVM...")
svm = LinearSVC()
svm.fit(Xtr_s, ytr)

pred_svm = svm.predict(Xte_s)

results["LinearSVM"] = {
    "accuracy": accuracy_score(yte, pred_svm),
    "auc": None   # LinearSVC has no probability estimates
}

# ==================================================
# 3️⃣ k-NN (Distance-based similarity)
# ==================================================
print("Training k-NN...")
knn = KNeighborsClassifier(
    n_neighbors=7,
    metric="euclidean"
)
knn.fit(Xtr_l2, ytr)

pred_knn = knn.predict(Xte_l2)
proba_knn = knn.predict_proba(Xte_l2)[:, 1]

results["kNN"] = {
    "accuracy": accuracy_score(yte, pred_knn),
    "auc": roc_auc_score(yte, proba_knn)
}

# ----------------------------
# Save results
# ----------------------------
with open(os.path.join(LOG_DIR, "embedding_baseline_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ----------------------------
# Print summary
# ----------------------------
print("\n=== Embedding Baseline Results ===")
for model, metrics in results.items():
    print(
        f"{model:>18} | "
        f"Acc: {metrics['accuracy']:.4f} | "
        f"AUC: {metrics['auc']}"
    )
