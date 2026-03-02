import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time
from multiprocessing import Pool, cpu_count

from src.utils.load_data import load_data
from src.IQL.models.fixed_memory_iqc import FixedMemoryIQC
from src.IQL.backends.hardware_native import HardwareNativeBackend
from src.utils.paths import load_paths
from src.utils.seed import set_seed

def train_and_eval_k(k, X_train, y_train, X_test, y_test, models_dir):
    """Worker function to train and evaluate a single K value."""
    print(f"\n--- Starting Training K={k} ---")
    start_time = time.time()
    
    # Initialize and train model
    # Note: HardwareNativeBackend and models usually need to be initialized inside the process
    # to avoid pickling issues, especially with quantum simulators/hardware backends.
    model = FixedMemoryIQC(K=k, eta=0.1, backend=HardwareNativeBackend())
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    train_time = time.time() - start_time
    print(f"✅ Finished K={k} | Test Accuracy: {acc:.4f} | Time: {train_time:.2f}s")

    # Save model
    model_name = f"fixed_memory_iqc_k{k}.pkl"
    model_path = os.path.join(models_dir, model_name)
    model.save(model_path)
    print(f"💾 Saved K={k} model to {model_path}")
    
    return k, acc

def main():
    # -------------------------------------------------
    # Load paths and Set Seed
    # -------------------------------------------------
    set_seed()
    BASE_ROOT, PATHS = load_paths()

    OUT_DIR = os.path.join(BASE_ROOT, "results", "fixed_memory_iqc_sweep")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    MODELS_DIR = os.path.join(OUT_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = load_data("polar")

    # Quantum-safe normalization (defensive)
    X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)

    # -------------------------------------------------
    # Sweep Setup
    # -------------------------------------------------
    # List of K values to sweep over 
    k_values = [10]#[i for i in range(1,20)] 
    
    # Use 3 workers or max CPU count, whichever is smaller
    num_workers = min(3, cpu_count())
    
    print(f"🚀 Starting K sweep for FixedMemoryIQC: {k_values}")
    print(f"⚡ Running with {num_workers} parallel workers...")
    
    overall_start_time = time.time()
    
    results = []
    # Create arguments for the worker pool
    pool_args = [(k, X_train, y_train, X_test, y_test, MODELS_DIR) for k in k_values]
    
    # Run multiprocessing pool
    with Pool(processes=num_workers) as pool:
        # Using starmap to pass multiple arguments to the worker function
        results = pool.starmap(train_and_eval_k, pool_args)

    # Sort results by K to ensure consistent plotting
    results.sort(key=lambda x: x[0])
    
    # Unpack sorted results
    sorted_k_values = [res[0] for res in results]
    accuracies = [res[1] for res in results]

    total_time = time.time() - overall_start_time
    print(f"\n🎉 All K sweeps completed in {total_time:.2f}s ({total_time/60:.2f} mins)")

    # -------------------------------------------------
    # Plotting
    # -------------------------------------------------
    plot_path = os.path.join(OUT_DIR, "k_vs_accuracy.png")
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_k_values, accuracies, marker='o', linestyle='-', color='b', label='Test Accuracy')
    plt.title("FixedMemoryIQC: K vs Accuracy")
    plt.xlabel("K (Memory Size)")
    plt.ylabel("Test Accuracy")
    plt.xticks(sorted_k_values)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 K vs Accuracy plot saved to: {plot_path}")

    # Optional: Save results summary to a text file
    summary_path = os.path.join(OUT_DIR, "sweep_results.txt")
    with open(summary_path, "w") as f:
        f.write("K\tAccuracy\n")
        for k, acc in zip(sorted_k_values, accuracies):
            f.write(f"{k}\t{acc:.4f}\n")
    print(f"📝 Sweep results summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
