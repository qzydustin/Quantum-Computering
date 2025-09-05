#!/usr/bin/env python3
"""
Quantum Algorithm Main Program
Unified interface for calling Grover, QAOA, and Simon quantum algorithms
"""

from grover_algorithm import grover_algorithm
from qaoa_algorithm import qaoa_algorithm
from simon_algorithm import simon_algorithm

def main():
    """Main function: Demonstrate three quantum algorithms"""
    print("=" * 50)
    print("Quantum Algorithm Demonstration Program")
    print("=" * 50)
    
    # 1. Grover algorithm demonstration
    print("\n1. Grover Search Algorithm")
    print("-" * 30)
    grover_qc = grover_algorithm(n_qubits=3)
    print(f"Grover circuit depth: {grover_qc.depth()}")
    print(f"Number of qubits: {grover_qc.num_qubits}")
    print(f"Number of classical bits: {grover_qc.num_clbits}")
    
    # 2. QAOA algorithm demonstration
    print("\n2. QAOA Quantum Approximate Optimization Algorithm")
    print("-" * 30)
    qaoa_qc = qaoa_algorithm(n_qubits=4, p=2)
    print(f"QAOA circuit depth: {qaoa_qc.depth()}")
    print(f"Number of qubits: {qaoa_qc.num_qubits}")
    print(f"Number of classical bits: {qaoa_qc.num_clbits}")
    
    # 3. Simon algorithm demonstration
    print("\n3. Simon Algorithm")
    print("-" * 30)
    simon_qc = simon_algorithm(n_qubits=3)
    print(f"Simon circuit depth: {simon_qc.depth()}")
    print(f"Number of qubits: {simon_qc.num_qubits}")
    print(f"Number of classical bits: {simon_qc.num_clbits}")
    
    print("\n" + "=" * 50)
    print("All algorithm demonstrations completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()
