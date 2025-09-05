from qiskit import QuantumCircuit
import numpy as np

def simon_algorithm(n_qubits=5, secret_string=None):
    """
    Create Simon algorithm circuit
    
    Simon algorithm is used to find hidden linear structures, providing exponential
    speedup compared to classical algorithms. Given a function f: {0,1}^n → {0,1}^n,
    if there exists a non-zero string s such that f(x) = f(x ⊕ s) for all x,
    Simon algorithm can find this hidden string s.
    
    Args:
        n_qubits: Number of qubits (size of each register), default 5 (total 10 qubits)
        secret_string: Hidden string, if None then randomly generated
        
    Returns:
        QuantumCircuit: Simon algorithm circuit
    """
    if secret_string is None:
        # Randomly generate hidden string (ensure it's not all zeros)
        secret_string = np.random.choice([0, 1], n_qubits)
        while np.all(secret_string == 0):
            secret_string = np.random.choice([0, 1], n_qubits)
    
    # Need two registers, each with n_qubits qubits
    qc = QuantumCircuit(2 * n_qubits, n_qubits)
    
    # Step 1: Create uniform superposition on first register
    for i in range(n_qubits):
        qc.h(i)
    
    # Step 2: Implement Oracle function
    # Here we implement a simple Oracle: for input x, output f(x) = x (when x < s)
    # or f(x) = x ⊕ s (when x >= s)
    
    # Simplified Oracle implementation: for demonstration purposes
    # In practical applications, Oracle implementation depends on specific problems
    
    # Copy information from first register to second register
    for i in range(n_qubits):
        qc.cx(i, n_qubits + i)
    
    # Apply conditional operations based on hidden string
    for i in range(n_qubits):
        if secret_string[i] == 1:
            # Apply conditional operation to the i-th qubit of first register
            qc.cx(i, n_qubits + (i + 1) % n_qubits)
    
    # Step 3: Apply Hadamard gates on first register
    for i in range(n_qubits):
        qc.h(i)
    
    # Step 4: Measure first register
    for i in range(n_qubits):
        qc.measure(i, i)
    
    return qc

if __name__ == "__main__":
    # Test code
    qc = simon_algorithm(n_qubits=3)
    print("Simon Algorithm Circuit:")
    print(qc)
