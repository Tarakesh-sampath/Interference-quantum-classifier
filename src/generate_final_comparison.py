import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from src.utils.paths import load_paths
from src.utils.seed import set_seed
from src.utils.load_data import load_data
from src.IQL.models.fixed_memory_iqc import FixedMemoryIQC
from src.IQL.models.static_isdo_model import StaticISDOModel
from src.IQL.backends.exact import ExactBackend
from src.IQL.backends.hardware_native import HardwareNativeBackend

def main():
    # ----------------------------
    # 0. Setup
    # ----------------------------
    set_seed(42)
    BASE_ROOT, PATHS = load_paths()
    RESULTS_FILE = os.path.join(BASE_ROOT, "results", "final_comparison_results.json")
    
    # ----------------------------
    # 1. Load Data
    # ----------------------------
    print("Loading data...")
    # Xtr, Xte: raw pre-normalized float64 embeddings
    # ytr_bin, yte_bin: binary labels (0, 1)
    # ytr_pol, yte_pol: polar labels (-1, 1)
    Xtr, Xte, ytr_bin, yte_bin, ytr_pol, yte_pol = load_data("all")
    
    final_results = {}

    # ----------------------------
    # 2. Classical Models
    # ----------------------------
    print("\nEvaluating Classical Models...")
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, n_jobs=-1)
    lr.fit(Xtr, ytr_bin)
    final_results["LogisticRegression"] = accuracy_score(yte_bin, lr.predict(Xte))
    print(f"  Logistic Regression: {final_results['LogisticRegression']:.4f}")

    # Linear SVM
    svm = LinearSVC(max_iter=2000)
    svm.fit(Xtr, ytr_bin)
    final_results["LinearSVM"] = accuracy_score(yte_bin, svm.predict(Xte))
    print(f"  Linear SVM:          {final_results['LinearSVM']:.4f}")

    # k-NN (k=5)
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(Xtr, ytr_bin)
    final_results["kNN"] = accuracy_score(yte_bin, knn.predict(Xte))
    print(f"  k-NN (k=5):          {final_results['kNN']:.4f}")

    # ----------------------------
    # 3. Quantum Models (IQC, K=2)
    # ----------------------------
    print("\nEvaluating IQC Models (K=2)...")
    K_val = 2

    # IQC Exact
    print("  Evaluating IQC Exact...")
    iqc_exact = FixedMemoryIQC(K=K_val, backend=ExactBackend())
    iqc_exact.fit(Xtr, ytr_pol)
    final_results["IQC_Exact"] = accuracy_score(yte_pol, iqc_exact.predict(Xte))
    print(f"    IQC Exact Acc:     {final_results['IQC_Exact']:.4f}")

    # IQC Hardware Native (1024 shots)
    print("Evaluating IQC Hardware Native...")
    # Using subsample for hardware simulation speed
    #EVAL_SAMPLES = 500
    #Xte_sub = Xte[:EVAL_SAMPLES]
    #yte_pol_sub = yte_pol[:EVAL_SAMPLES]
    
    IQL_PATH = os.path.join(BASE_ROOT, "results", "fixed_memory_iqc_sweep", "models", "fixed_memory_iqc_k2.pkl")
    iql_hw = FixedMemoryIQC.load(IQL_PATH)
    final_results["IQC_HardwareNative"] = accuracy_score(yte_pol, iql_hw.predict(Xte))
    print(f"    IQC HW Native Acc: {final_results['IQC_HardwareNative']:.4f}")

    # ----------------------------
    # 4. Static ISDO (K=2)
    # ----------------------------
    print("\nEvaluating Static ISDO (K=2)...")
    isdo = StaticISDOModel(K=K_val)
    isdo.fit(Xtr, ytr_bin)
    final_results["ISDO_K"] = accuracy_score(yte_bin, isdo.predict(Xte))
    print(f"  Static ISDO Acc:     {final_results['ISDO_K']:.4f}")

    # ----------------------------
    # 5. Save Results
    # ----------------------------
    with open(RESULTS_FILE, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✅ Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
