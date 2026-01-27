import numpy as np
from src.IQL.learning.class_state import ClassState
from src.IQL.learning.memory_bank import MemoryBank
from src.IQL.backends.exact import ExactBackend
from src.IQL.models.online_perceptron import OnlinePerceptron
from src.IQL.models.winner_take_all import WinnerTakeAll
from src.IQL.models.adaptive_memory import AdaptiveMemory

def test_consistency():
    print("Running consistency tests...")
    
    # 1. Backend
    backend = ExactBackend()
    
    # 2. ClassState
    vec = np.array([1, 0, 0, 0], dtype=np.complex128)
    cs = ClassState(vec, backend)
    print("ClassState initialized.")
    
    psi = np.array([1, 0, 0, 0], dtype=np.complex128)
    score = cs.score(psi)
    print(f"ClassState score: {score}")
    assert np.isclose(score, 1.0)
    
    # 3. MemoryBank
    mb = MemoryBank([cs])
    print("MemoryBank initialized.")
    scores = mb.scores(psi)
    print(f"MemoryBank scores: {scores}")
    assert np.isclose(scores[0], 1.0)
    
    # 4. Models
    # OnlinePerceptron
    op = OnlinePerceptron(cs, eta=0.1)
    y_hat, s, updated = op.step(psi, 1)
    print(f"OnlinePerceptron step: y_hat={y_hat}, s={s}, updated={updated}")
    
    # WinnerTakeAll
    wta = WinnerTakeAll(mb, eta=0.1, backend=backend)
    y_hat, idx, updated = wta.step(psi, 1)
    print(f"WinnerTakeAll step: y_hat={y_hat}, idx={idx}, updated={updated}")
    
    # AdaptiveMemory
    am = AdaptiveMemory(mb, eta=0.1, backend=backend)
    margin, spawned = am.step(psi, 1)
    print(f"AdaptiveMemory step: margin={margin}, spawned={spawned}")
    
    print("All basic consistency tests passed!")

if __name__ == "__main__":
    test_consistency()
