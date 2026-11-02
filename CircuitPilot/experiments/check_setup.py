import matplotlib
import numpy as np
import pandas as pd
import pennylane as qml
import sklearn
from pennylane import numpy as pnp

print("PennyLane:", qml.__version__)
print("NumPy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Pandas:", pd.__version__)

# 1. Bell state: H then CNOT should give |00> and |11> with 50% each
dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev)
def bell_state():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0, 1])


print("\nBell state probabilities [00, 01, 10, 11]:", np.round(bell_state(), 4))
print(qml.draw(bell_state)())

# 2. Gradient check: for RY(theta)|0>, <Z> = cos(theta), so d<Z>/dtheta = -sin(theta)
dev_1q = qml.device("default.qubit", wires=1)


@qml.qnode(dev_1q)
def ry_expval(theta):
    qml.RY(theta, wires=0)
    return qml.expval(qml.PauliZ(0))


theta = pnp.array(0.5, requires_grad=True)
print("\n<Z> after RY(0.5):", round(float(ry_expval(theta)), 4), " expected:", round(np.cos(0.5), 4))
print("Gradient:          ", round(float(qml.grad(ry_expval)(theta)), 4), " expected:", round(-np.sin(0.5), 4))

# 3. Noise check: depolarizing noise with probability p on |0> gives <Z> = 1 - 4p/3
noisy_dev = qml.device("default.mixed", wires=1)


@qml.qnode(noisy_dev)
def noisy_zero(p):
    qml.DepolarizingChannel(p, wires=0)
    return qml.expval(qml.PauliZ(0))


p = 0.1
print("\nNoisy <Z> with p=0.1:", round(float(noisy_zero(p)), 4), " expected:", round(1 - 4 * p / 3, 4))
