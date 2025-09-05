from qiskit import QuantumCircuit
import numpy as np

def qaoa_algorithm(n_qubits=10, p=2, gamma=None, beta=None):
    """
    Create QAOA (Quantum Approximate Optimization Algorithm) circuit
    
    QAOA is a quantum algorithm for solving combinatorial optimization problems.
    This implementation creates a QAOA circuit for solving the Max-Cut problem.
    
    Args:
        n_qubits: Number of qubits, default 10 suitable for real device testing
        p: Number of QAOA layers
        gamma: List of parameters for problem Hamiltonian
        beta: List of parameters for mixer Hamiltonian
        
    Returns:
        QuantumCircuit: QAOA circuit
    """
    if gamma is None:
        gamma = [np.pi/4] * p
    if beta is None:
        beta = [np.pi/2] * p
    
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Step 1: Initialization - Create uniform superposition
    for i in range(n_qubits):
        qc.h(i)
    
    # Step 2: QAOA layers
    for layer in range(p):
        # Apply problem Hamiltonian (Max-Cut problem)
        # For Max-Cut, we apply ZZ rotations between adjacent qubits
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                # ZZ rotation gate: exp(-i * gamma * Z_i * Z_j)
                qc.cx(i, j)
                qc.rz(2 * gamma[layer], j)
                qc.cx(i, j)
        
        # Apply mixer Hamiltonian
        for i in range(n_qubits):
            qc.rx(2 * beta[layer], i)
    
    # Measure qubits only, don't create additional classical bits
    for i in range(n_qubits):
        qc.measure(i, i)
    return qc

if __name__ == "__main__":
    # Test code
    qc = qaoa_algorithm(n_qubits=4, p=2)
    print("QAOA Algorithm Circuit:")
    print(qc)
