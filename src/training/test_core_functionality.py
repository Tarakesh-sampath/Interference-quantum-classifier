# tests/test_core_functionality.py
"""
Comprehensive test suite for quantum classifier core functionality.
Run with: python -m pytest tests/test_core_functionality.py -v
"""
import pytest
import numpy as np
from src.IQL.learning.class_state import ClassState
from src.IQL.learning.memory_bank import MemoryBank
from src.IQL.backends.exact import ExactBackend
from src.IQL.regimes.regime2_online import OnlinePerceptron
from src.IQL.regimes.regime3a_wta import WinnerTakeAll
from src.utils.label_utils import binary_to_polar, polar_to_binary, ensure_polar

class TestLabelConversions:
    """Test label conversion utilities."""
    
    def test_binary_to_polar(self):
        binary = np.array([0, 1, 0, 1])
        polar = binary_to_polar(binary)
        expected = np.array([-1, 1, -1, 1])
        np.testing.assert_array_equal(polar, expected)
    
    def test_polar_to_binary(self):
        polar = np.array([-1, 1, -1, 1])
        binary = polar_to_binary(polar)
        expected = np.array([0, 1, 0, 1])
        np.testing.assert_array_equal(binary, expected)
    
    def test_round_trip(self):
        binary = np.array([0, 1, 0, 1])
        polar = binary_to_polar(binary)
        back_to_binary = polar_to_binary(polar)
        np.testing.assert_array_equal(binary, back_to_binary)
    
    def test_ensure_polar_from_binary(self):
        binary = np.array([0, 1])
        polar = ensure_polar(binary)
        np.testing.assert_array_equal(polar, np.array([-1, 1]))
    
    def test_ensure_polar_from_polar(self):
        polar = np.array([-1, 1])
        result = ensure_polar(polar)
        np.testing.assert_array_equal(result, polar)

class TestClassState:
    """Test ClassState functionality."""
    
    def test_initialization(self):
        backend = ExactBackend()
        vec = np.array([1, 0, 0, 0], dtype=np.complex128)
        cs = ClassState(vec, backend, label=1)
        
        assert cs.label == 1
        assert np.isclose(np.linalg.norm(cs.vector), 1.0)
    
    def test_normalization(self):
        backend = ExactBackend()
        vec = np.array([3, 4, 0, 0], dtype=np.complex128)
        cs = ClassState(vec, backend)
        
        # Should be normalized
        assert np.isclose(np.linalg.norm(cs.vector), 1.0)
    
    def test_score_orthogonal(self):
        backend = ExactBackend()
        chi = np.array([1, 0, 0, 0], dtype=np.complex128)
        psi = np.array([0, 1, 0, 0], dtype=np.complex128)
        
        cs = ClassState(chi, backend)
        score = cs.score(psi)
        
        assert np.isclose(score, 0.0, atol=1e-10)
    
    def test_score_parallel(self):
        backend = ExactBackend()
        vec = np.array([1, 0, 0, 0], dtype=np.complex128)
        
        cs = ClassState(vec, backend)
        score = cs.score(vec)
        
        assert np.isclose(score, 1.0, atol=1e-10)

class TestMemoryBank:
    """Test MemoryBank functionality."""
    
    def test_initialization(self):
        backend = ExactBackend()
        cs1 = ClassState(np.array([1, 0, 0, 0], dtype=np.complex128), backend, label=0)
        cs2 = ClassState(np.array([0, 1, 0, 0], dtype=np.complex128), backend, label=1)
        
        mb = MemoryBank([cs1, cs2])
        assert len(mb.class_states) == 2
    
    def test_add_memory(self):
        backend = ExactBackend()
        cs = ClassState(np.array([1, 0, 0, 0], dtype=np.complex128), backend)
        mb = MemoryBank([cs])
        
        new_vec = np.array([0, 1, 0, 0], dtype=np.complex128)
        mb.add_memory(new_vec, backend, label=1)
        
        assert len(mb.class_states) == 2
        assert mb.class_states[1].label == 1
    
    def test_remove_memory(self):
        backend = ExactBackend()
        cs1 = ClassState(np.array([1, 0, 0, 0], dtype=np.complex128), backend)
        cs2 = ClassState(np.array([0, 1, 0, 0], dtype=np.complex128), backend)
        cs3 = ClassState(np.array([0, 0, 1, 0], dtype=np.complex128), backend)
        
        mb = MemoryBank([cs1, cs2, cs3])
        mb.remove(1)
        
        assert len(mb.class_states) == 2
    
    def test_winner(self):
        backend = ExactBackend()
        cs1 = ClassState(np.array([1, 0, 0, 0], dtype=np.complex128), backend)
        cs2 = ClassState(np.array([0, 1, 0, 0], dtype=np.complex128), backend)
        
        mb = MemoryBank([cs1, cs2])
        
        # Test with state close to cs1
        psi = np.array([0.9, 0.1, 0, 0], dtype=np.complex128)
        psi /= np.linalg.norm(psi)
        
        idx, score = mb.winner(psi)
        assert idx == 0  # Should select cs1

class TestOnlinePerceptron:
    """Test OnlinePerceptron regime."""
    
    def test_training_convergence(self):
        backend = ExactBackend()
        
        # Simple linearly separable data
        X = np.array([
            [1, 0, 0, 0],
            [0.9, 0.1, 0, 0],
            [0, 1, 0, 0],
            [0, 0.9, 0.1, 0],
        ], dtype=np.complex128)
        
        y = np.array([1, 1, -1, -1])
        
        # Initialize with random state
        chi0 = np.array([0.5, 0.5, 0, 0], dtype=np.complex128)
        chi0 /= np.linalg.norm(chi0)
        
        cs = ClassState(chi0, backend)
        trainer = OnlinePerceptron(cs, eta=0.1)
        
        acc = trainer.fit(X, y)
        
        # Should achieve reasonable accuracy on this simple problem
        assert acc >= 0.5
    
    def test_save_load(self):
        import tempfile
        import os
        
        backend = ExactBackend()
        chi = np.array([1, 0, 0, 0], dtype=np.complex128)
        cs = ClassState(chi, backend)
        trainer = OnlinePerceptron(cs, eta=0.1)
        
        # Train a bit
        X = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.complex128)
        y = np.array([1, -1])
        trainer.fit(X, y)
        
        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            trainer.save(temp_path)
            
            # Load
            loaded_trainer = OnlinePerceptron.load(temp_path)
            
            # Check attributes
            assert loaded_trainer.eta == trainer.eta
            assert loaded_trainer.num_updates == trainer.num_updates
            assert len(loaded_trainer.history["scores"]) == len(trainer.history["scores"])
        finally:
            os.unlink(temp_path)

def test_integration():
    """Integration test for full pipeline."""
    from src.IQL.regimes.regime3a_wta import WinnerTakeAll
    
    backend = ExactBackend()
    
    # Create simple data
    X = np.array([
        [1, 0, 0, 0],
        [0.9, 0.1, 0, 0],
        [0, 1, 0, 0],
        [0.1, 0.9, 0, 0],
    ], dtype=np.complex128)
    y = np.array([1, 1, -1, -1])
    
    # Initialize memory bank
    cs1 = ClassState(np.array([1, 0, 0, 0], dtype=np.complex128), backend, label=1)
    cs2 = ClassState(np.array([0, 1, 0, 0], dtype=np.complex128), backend, label=-1)
    mb = MemoryBank([cs1, cs2])
    
    # Train with WTA
    wta = WinnerTakeAll(mb, eta=0.1, backend=backend)
    acc = wta.fit(X, y)
    
    # Predict
    predictions = wta.predict(X)
    
    assert len(predictions) == len(X)
    assert acc >= 0.5  # Should do reasonably well

if __name__ == "__main__":
    pytest.main([__file__, "-v"])