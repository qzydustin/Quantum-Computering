from qiskit import QuantumCircuit
import numpy as np

def grover_algorithm(n_qubits=3, marked_states=None):
    """
    Create Grover search algorithm circuit
    
    Grover algorithm is used to search for specific items in an unordered database,
    providing quadratic speedup. For a database with N items, classical algorithms
    require O(N) queries, while Grover algorithm only needs O(√N).
    
    Args:
        n_qubits: Number of qubits (search space size is 2^n_qubits), 
                  default 10 suitable for real device testing
        marked_states: List of marked states, if None then defaults to marking the last state
        
    Returns:
        QuantumCircuit: Grover search algorithm circuit
    """
    if marked_states is None:
        marked_states = [2**n_qubits - 1]  # Default to marking the last state
    
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Step 1: Initialization - Create uniform superposition
    for i in range(n_qubits):
        qc.h(i)
    
    # Calculate optimal number of iterations
    N = 2**n_qubits  # Search space size
    M = len(marked_states)  # Number of target states
    optimal_iterations = int(np.floor(np.pi / 4 * np.sqrt(N / M)))
    
    # Step 2: Grover iteration
    for iteration in range(optimal_iterations):
        
        # Oracle: Mark target states
        for marked_state in marked_states:
            # Convert marked state to binary representation
            binary_state = format(marked_state, f'0{n_qubits}b')
            
            # Note: In Qiskit, qubit indices and binary strings are opposite
            # Qubit 0 corresponds to least significant bit (rightmost), 
            # qubit n-1 corresponds to most significant bit (leftmost)
            # So we need to reverse the binary string to correctly apply X gates
            binary_state = binary_state[::-1]  # Reverse string
            
            # Apply X gates to bits that are not 1
            for i, bit in enumerate(binary_state):
                if bit == '0':
                    qc.x(i)
            
            # Multi-control Z gate (phase marking)
            if n_qubits == 1:
                qc.z(0)
            elif n_qubits == 2:
                qc.cz(0, 1)
            else:
                # Use multi-control Z gate to implement phase flip
                qc.h(n_qubits - 1)
                qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
                qc.h(n_qubits - 1)
            
            # Undo X gates
            for i, bit in enumerate(binary_state):
                if bit == '0':
                    qc.x(i)
        
        # Diffusion operator = H^⊗n · Z_0 · H^⊗n
        
        # First step: Apply Hadamard gates
        for i in range(n_qubits):
            qc.h(i)
        
        # Second step: Apply phase flip to |0⟩ state (Z_0)
        # Equivalent to applying X to all bits, then multi-control Z, then undo X
        for i in range(n_qubits):
            qc.x(i)
        
        # Multi-control Z gate
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        
        # Undo X gates
        for i in range(n_qubits):
            qc.x(i)
        
        # Third step: Apply Hadamard gates again
        for i in range(n_qubits):
            qc.h(i)
    
    # Final measurement
    for i in range(n_qubits):
        qc.measure(i, i)
    return qc

if __name__ == "__main__":
    # Test code
    qc = grover_algorithm(n_qubits=3)
    print("Grover Algorithm Circuit:")
    print(qc)
