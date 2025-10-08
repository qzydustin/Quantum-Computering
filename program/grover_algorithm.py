from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np

def grover_algorithm(n_qubits=3, marked_states=None):
    """
    Create a Grover search circuit following IBM's reference implementation with
    an ancilla prepared in |->, an oracle via phase kickback, and the standard
    diffusion operator on the input register.

    Reference: IBM Quantum Learning – Grover's algorithm
    https://quantum.cloud.ibm.com/learning/en/courses/utility-scale-quantum-computing/grovers-algorithm

    Args:
        n_qubits: Number of search qubits (the search space has size 2^n_qubits).
        marked_states: Iterable of integers representing the marked basis states.
            If None, defaults to marking the last basis state (2^n - 1).

    Returns:
        QuantumCircuit: The Grover circuit measuring only the input register.
    """
    if marked_states is None:
        marked_states = [2 ** n_qubits - 1]

    # Build circuit with named registers so wires are labeled in diagrams
    inp = QuantumRegister(n_qubits, "inp")
    anc = QuantumRegister(1, "anc")  # anc[0] is the ancilla qubit
    creg = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(inp, anc, creg)

    # 1) Initialization: create uniform superposition on inputs; ancilla to |->
    for q in range(n_qubits):
        qc.h(inp[q])
    qc.x(anc[0])
    qc.h(anc[0])
    qc.barrier()

    # Helper: oracle via phase kickback on ancilla for a single marked bitstring
    def apply_oracle_for_bits(bitstring: str) -> None:
        # Align q0 with the rightmost (LSB) bit in Qiskit's little-endian ordering
        bits = bitstring[::-1]
        # Flip zeros so controls fire when state == target
        for idx, bit in enumerate(bits):
            if bit == '0':
                qc.x(inp[idx])
        # Multi-controlled X onto ancilla implements a phase flip via |->
        if n_qubits == 1:
            qc.cx(inp[0], anc[0])
        else:
            qc.mcx([inp[i] for i in range(n_qubits)], anc[0])
        # Uncompute the preparation
        for idx, bit in enumerate(bits):
            if bit == '0':
                qc.x(inp[idx])
        qc.barrier()

    # Helper: standard diffusion operator on input qubits
    def apply_diffusion() -> None:
        # H on inputs
        for q in range(n_qubits):
            qc.h(inp[q])
        # X on inputs
        for q in range(n_qubits):
            qc.x(inp[q])
        # Implement multi-controlled Z about |0...0>
        if n_qubits == 1:
            qc.z(inp[0])
        elif n_qubits == 2:
            qc.cz(inp[0], inp[1])
        else:
            qc.h(inp[-1])
            qc.mcx([inp[i] for i in range(n_qubits - 1)], inp[-1])
            qc.h(inp[-1])
        # Uncompute X and finish H
        for q in range(n_qubits):
            qc.x(inp[q])
        for q in range(n_qubits):
            qc.h(inp[q])
        qc.barrier()

    # Optimal number of Grover iterations ≈ floor(pi/4 * sqrt(N / M))
    N = 2 ** n_qubits
    M = max(1, len(marked_states))
    num_iterations = int(np.floor((np.pi / 4.0) * np.sqrt(N / M)))

    # 2) Grover iterations: Oracle then Diffusion
    for _ in range(num_iterations):
        for state in marked_states:
            bitstring = format(state, f'0{n_qubits}b')
            apply_oracle_for_bits(bitstring)
        apply_diffusion()

    # 3) Clear ancilla back to |0>
    qc.h(anc[0])
    qc.x(anc[0])
    qc.barrier()

    # 4) Measure only input qubits
    for i in range(n_qubits):
        qc.measure(inp[i], creg[i])

    return qc

if __name__ == "__main__":
    qc = grover_algorithm(n_qubits=3)
    print("Grover Algorithm Circuit:")
    print(qc)
