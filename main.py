
from ibm_quantum_connector import QuantumServiceManager
from quantum_executor import QuantumExecutor
from quantum_utils import QuantumResultManager, save_theoretical_test_results
from program.grover_algorithm import grover_algorithm
from tools.generate_charts import plot_comprehensive_analysis_from_json

def get_test_circuit(name, **kwargs):
    """
    Get quantum algorithm circuit
    
    Args:
        name: Algorithm name
        **kwargs: Algorithm parameters
        
    Returns:
        QuantumCircuit: Quantum algorithm circuit
    """
    circuit_catalog = {
        "grover": grover_algorithm
    }
    
    if name not in circuit_catalog:
        raise ValueError(f"Unknown algorithm name: {name}. Available algorithms: {list(circuit_catalog.keys())}")
    
    return circuit_catalog[name](**kwargs)

def list_available_circuits():
    """List all available quantum algorithms"""
    return ["grover"]
import json
import numpy as np

def initialize_executor():
    """Initialize quantum executor"""
    print("Initializing quantum service manager, please wait...")
    # Initialize quantum service manager
    service_manager = QuantumServiceManager()
    
    print("正在连接到 IBM Quantum 服务，请稍候...")
    # 连接到 IBM Quantum 服务
    if not service_manager.connect():
        return None
    
    print("Selecting backend device...")
    # 选择后端设备（用于校验配置是否可用）
    backend = service_manager.select_backend()
    if not backend:
        return None

    print("正在创建量子执行器，请稍候...")
    # 创建量子执行器（使用服务对象，后端从配置读取并解析为对象）
    executor = QuantumExecutor(service=service_manager.service)
    print("量子执行器初始化完成。")
    return executor

    
if __name__ == "__main__":
    # 初始化执行器（连接服务，仅用于获取backend以构建噪声模型）
    executor = initialize_executor()
    if executor is None:
        print("无法初始化量子执行器，程序退出")
        exit(1)

    from delta_debug import run_delta_debug_on_grover
    n_qubits = 4
    marked_states = [2**n_qubits - 1]
    print(f"开始Delta Debugging分析(噪声模拟): {n_qubits}量子比特, 目标状态: {marked_states}")
    result, report_file = run_delta_debug_on_grover(executor, n_qubits, marked_states)
    print(f"\nDelta Debugging完成，报告已保存: {report_file}")
    plot_comprehensive_analysis_from_json(report_file)
