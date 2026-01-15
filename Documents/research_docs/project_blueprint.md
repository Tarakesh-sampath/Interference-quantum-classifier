# Accelerated Research Project Blueprint (8-Week Roadmap)

This revised plan compresses the research into a high-intensity **8-week cycle**, focusing on the critical implementation of the measurement-free quantum classifier.

---

## Part 1: Foundation & Architecture (Weeks 1-2)
**Goal**: Rapid setup and interface design.
- **Week 1: Infrastructure & Data**: 
    - Configure Qiskit/PyTorch environment.
    - Set up a **subset** data loader for PathCamelyon (to speed up iteration).
    - Implement a pre-trained CNN feature extractor (e.g., ResNet18) instead of training from scratch.
- **Week 2: Quantum-Classical Interface**: 
    - Implement Amplitude Encoding for 8D/16D features.
    - Prototype the SWAP-test circuit and verify basic state overlap logic.

## Part 2: Implementation & Hybrid Training (Weeks 3-5)
**Goal**: Build the core and optimize prototypes.
- **Week 3: Circuit Optimization**: 
    - Minimize gate depth for NISQ feasibility (target <50 gates if possible).
    - Implement noisy simulation environment.
- **Week 4-5: Joint Optimization**: 
    - Execute hybrid training loops using the Parameter-Shift rule.
    - Focus on optimizing class prototypes to maximize inter-class fidelity distance.
    - Monitor for training stability in a shorter epoch window.

## Part 3: Validation & Reporting (Weeks 6-8)
**Goal**: Prove innovation and finalize documentation.
- **Week 6: Performance Evaluation**: 
    - Calculate Accuracy, F1-Score, and AUC-ROC on the test set.
    - Run primary comparison against a standard VQC baseline.
- **Week 7: Robustness & Noise Study**: 
    - Test the measurement-free advantage by simulating hardware noise.
    - Conduct a single hardware run (IBM Quantum) if possible.
- **Week 8: Final Synthesis**: 
    - Finalize the technical report/manuscript.
    - Prepare visualizations and code documentation for handover.

---

## Streamlining Strategy
- **Pre-trained Backbones**: Use pre-trained weights to skip weeks of classical training.
- **Sub-sampling**: Use a balanced subset of PatchCamelyon for training to reduce compute time.
- **Parallelization**: Design circuits while the data pipeline is being finalized.
- **Focus**: Prioritize "Proof of Concept" over "Scale" to meet the 8-week deadline.
