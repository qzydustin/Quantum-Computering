from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import json
from datetime import datetime

class QuantumDeltaDebugger:
    """
    量子电路Delta Debugging调试器
    使用DDMin算法找出导致目标态概率损失最严重的电路段
    """
    
    def __init__(self, executor, target_states, tolerance=0.02):
        """
        初始化Delta Debugger
        
        Args:
            executor: 量子执行器实例
            target_states: 目标状态列表
            tolerance: 概率损失容忍度
        """
        self.executor = executor
        self.target_states = target_states
        self.tolerance = tolerance
        self.ideal_sim = AerSimulator()
        self.debug_history = []
        self.test_count = 0
        self.logical_n_qubits = None  # number of data qubits to evaluate on
        
    def _ensure_measured(self, circ: QuantumCircuit) -> QuantumCircuit:
        """Ensure the circuit has exactly one set of measurements.

        - If the circuit already contains measure ops, return as-is.
        - If there are fewer classical bits than qubits (common for algorithmic
          circuits without classical regs), create a new circuit with matching
          classical bits, copy non-measure ops, then measure_all.
        - Otherwise, call measure_all once.
        """
        try:
            if any(inst.operation.name == "measure" for inst in circ.data):
                return circ

            if circ.num_clbits < circ.num_qubits:
                # Rebuild with enough classical bits and add a single measurement set
                new_qc = QuantumCircuit(circ.num_qubits, circ.num_qubits)
                for inst in circ.data:
                    if inst.operation.name != "measure":
                        new_qc.append(inst.operation, inst.qubits, inst.clbits)
                new_qc.measure_all()
                return new_qc
            else:
                circ.measure_all()
                return circ
        except Exception:
            # Fallback: ensure a minimal measurable circuit
            new_qc = QuantumCircuit(circ.num_qubits, max(circ.num_qubits, circ.num_clbits))
            for inst in getattr(circ, 'data', []):
                if getattr(inst.operation, 'name', None) != "measure":
                    new_qc.append(inst.operation, inst.qubits, inst.clbits)
            new_qc.measure_all()
            return new_qc

    def _bitstr_for_state(self, circuit: QuantumCircuit, state_int: int) -> str:
        """Return the bitstring key for counts matching q[i]->c[i] (measure_all case).

        For measure_all with direct q[i]→c[i], Qiskit's counts bitstring is in classical
        register order, which corresponds to reversing the natural binary order.
        """
        n = circuit.num_qubits
        return format(state_int, f'0{n}b')[::-1]

    def extract_circuit_segments(self, circuit):
        """
        将量子电路分解为可测试的段
        
        Args:
            circuit: 量子电路
            
        Returns:
            list: 电路段列表，每个段包含指令和位置信息
        """
        segments = []
        
        # 按层分组指令
        current_layer = []
        for i, instruction in enumerate(circuit.data):
            if instruction.operation.name == 'measure':
                continue  # 跳过测量指令
                
            current_layer.append({
                'instruction': instruction,
                'index': i,
                'operation': instruction.operation.name,
                'qubits': [circuit.find_bit(qubit).index for qubit in instruction.qubits],
                'params': getattr(instruction.operation, 'params', [])
            })
            
            # 检查是否应该结束当前层
            if self._should_end_layer(current_layer, instruction):
                if current_layer:
                    segments.append({
                        'instructions': current_layer.copy(),
                        'layer_id': len(segments),
                        'description': self._describe_layer(current_layer)
                    })
                    current_layer = []
        
        # 添加最后一层
        if current_layer:
            segments.append({
                'instructions': current_layer.copy(),
                'layer_id': len(segments),
                'description': self._describe_layer(current_layer)
            })
        
        return segments
    
    def _should_end_layer(self, current_layer, instruction):
        """判断是否应该结束当前层"""
        # 简单的分层策略：每5个指令或遇到控制门时分层
        if len(current_layer) >= 5:
            return True
        if instruction.operation.name in ['cx', 'cz', 'mcx', 'ccx']:
            return True
        return False
    
    def _describe_layer(self, layer):
        """描述层的内容"""
        ops = [inst['operation'] for inst in layer]
        op_counts = {}
        for op in ops:
            op_counts[op] = op_counts.get(op, 0) + 1
        
        description = ", ".join([f"{count}×{op}" for op, count in op_counts.items()])
        return f"Layer({description})"
    
    def build_circuit_without_segments(self, original_circuit, segments_to_exclude):
        """
        构建排除指定段的电路
        
        Args:
            original_circuit: 原始电路
            segments_to_exclude: 要排除的段的索引列表
            
        Returns:
            QuantumCircuit: 修改后的电路
        """
        # 创建空电路外壳（优先保留寄存器结构；测量稍后统一处理）
        try:
            new_circuit = original_circuit.copy_empty_like()
        except Exception:
            new_circuit = QuantumCircuit(original_circuit.num_qubits, original_circuit.num_clbits)

        # 计算要排除的指令索引（源自分段，不含测量）
        excluded_indices = set()
        for seg_idx in segments_to_exclude:
            if seg_idx < len(self.segments):
                for inst in self.segments[seg_idx]['instructions']:
                    excluded_indices.add(inst['index'])

        # 追加非排除的非测量指令；避免重复测量
        for i, (op, qargs, cargs) in enumerate(original_circuit.data):
            if getattr(op, 'name', None) == 'measure':
                continue
            if i not in excluded_indices:
                # 重新映射量子位到新电路上的对应位置
                try:
                    q_indices = [original_circuit.find_bit(q).index for q in qargs]
                    mapped_qargs = [new_circuit.qubits[idx] for idx in q_indices]
                except Exception:
                    # 回退：直接使用顺序映射（假定顺序一致）
                    mapped_qargs = [new_circuit.qubits[j] for j in range(len(qargs))]
                new_circuit.append(op, mapped_qargs, [])

        # 统一只测量一次：若设置了逻辑数据比特，只测前 n 个以避免把辅比特带入
        if self.logical_n_qubits is None:
            new_circuit = self._ensure_measured(new_circuit)
        else:
            # 确保有足够经典位
            if new_circuit.num_clbits < self.logical_n_qubits:
                tmp = QuantumCircuit(new_circuit.num_qubits, self.logical_n_qubits)
                for inst in new_circuit.data:
                    if getattr(inst.operation, 'name', None) != 'measure':
                        tmp.append(inst.operation, inst.qubits, inst.clbits)
                new_circuit = tmp
            # 添加前 n 个数据比特的测量
            for i in range(self.logical_n_qubits):
                new_circuit.measure(i, i)
        return new_circuit
    
    def evaluate_circuit(self, circuit, shots=4096):
        """
        评估电路的目标态概率（理想模拟 vs 带噪声模拟）
        
        Args:
            circuit: 量子电路
            shots: 测量次数
            
        Returns:
            tuple: (ideal_target_prob, noisy_target_prob, ideal_counts)
        """
        self.test_count += 1
        
        try:
            # 使用统一执行器跑理想模拟
            ideal_res = self.executor.run_circuit(circuit, execution_type="ideal_simulator", shots=shots)
            if not ideal_res.get('success'):
                raise RuntimeError(f"Ideal simulation failed: {ideal_res.get('error')}")
            ideal_counts = ideal_res.get('counts', {})
            # 将 counts 投影到逻辑数据比特空间（小端）
            n_data = self.logical_n_qubits or circuit.num_qubits
            from collections import defaultdict
            def _project_counts_to_data_qubits(counts, n_data_qubits):
                acc = defaultdict(int)
                for k, v in (counts or {}).items():
                    bits_le = k.replace(" ", "")[::-1]
                    key = bits_le[:n_data_qubits]
                    acc[key] += v
                return dict(acc)
            proj_ideal = _project_counts_to_data_qubits(ideal_counts, n_data)
            total_ideal = max(1, sum(proj_ideal.values()))
            ideal_target_prob = 0.0
            def _target_key_le(state_int, n_bits):
                return format(state_int, f'0{n_bits}b')
            for state in self.target_states:
                key = _target_key_le(state, n_data)
                ideal_target_prob += proj_ideal.get(key, 0) / total_ideal
            
            # 使用带噪声模拟器（从后端构建噪声）
            noisy_shots = max(1, shots // 2)
            noisy_target_prob = 0.0
            try:
                noisy_res = self.executor.run_circuit(circuit, execution_type="noisy_simulator", shots=noisy_shots)
                if noisy_res.get('success'):
                    noisy_counts = noisy_res.get('counts', {})
                    proj_noisy = _project_counts_to_data_qubits(noisy_counts, n_data)
                    total_noisy = max(1, sum(proj_noisy.values()))
                    for state in self.target_states:
                        key = _target_key_le(state, n_data)
                        noisy_target_prob += proj_noisy.get(key, 0) / total_noisy
                else:
                    print(f"带噪声模拟失败: {noisy_res.get('error')}")
            except Exception as ne:
                print(f"带噪声模拟异常: {ne}")
                noisy_target_prob = 0.0
            
            return ideal_target_prob, noisy_target_prob, ideal_counts
            
        except Exception as e:
            print(f"电路评估失败: {e}")
            return 0.0, 0.0, {}
    
    def ddmin(self, segments):
        """
        DDMin算法实现
        
        Args:
            segments: 电路段列表
            
        Returns:
            list: 导致问题的最小段集合
        """
        print(f"\n开始DDMin调试，共{len(segments)}个段...")
        
        # 评估完整电路
        full_circuit = self.build_circuit_without_segments(self.original_circuit, [])
        full_ideal_prob, full_noisy_prob, _ = self.evaluate_circuit(full_circuit)
        
        print(f"完整电路 - 理想概率: {full_ideal_prob:.4f}, 噪声概率: {full_noisy_prob:.4f}")
        
        # 评估空电路（只有初始化和测量）
        n_data = self.logical_n_qubits or self.original_circuit.num_qubits
        empty_circuit = QuantumCircuit(n_data, n_data)
        for i in range(n_data):
            empty_circuit.h(i)  # 初始化为均匀叠加态
            empty_circuit.measure(i, i)
        
        empty_ideal_prob, empty_noisy_prob, _ = self.evaluate_circuit(empty_circuit)
        print(f"空电路 - 理想概率: {empty_ideal_prob:.4f}, 噪声概率: {empty_noisy_prob:.4f}")
        
        # 计算目标概率损失
        target_loss = full_ideal_prob - full_noisy_prob
        print(f"目标概率损失: {target_loss:.4f}")
        
        if target_loss < self.tolerance:
            print("概率损失在容忍范围内，无需调试")
            return []
        
        # DDMin主循环
        candidates = list(range(len(segments)))
        n = 2  # 分割粒度
        iteration_count = 0
        
        while len(candidates) >= 2:
            iteration_count += 1
            print(f"\n{'='*50}")
            print(f"DDMin迭代 #{iteration_count}: {len(candidates)}个候选段, 分割粒度: {n}")
            print(f"当前基准损失: {target_loss:.4f}")
            print(f"容忍度: {self.tolerance:.4f}")
            print(f"{'='*50}")
            
            # 将候选段分割为n个子集
            subsets = self._split_into_subsets(candidates, n)
            any_reduction = False
            
            # 测试每个子集的补集
            for i, subset in enumerate(subsets):
                complement = [seg for seg in candidates if seg not in subset]
                
                if not complement:  # 跳过空补集
                    continue
                
                print(f"\n  测试子集{i+1}/{len(subsets)}：排除该子集（等价于保留其补集）")
                print(f"    被排除段: {subset}")
                print(f"    被保留段(补集): {complement}")
                
                # 构建排除当前子集的电路
                test_circuit = self.build_circuit_without_segments(self.original_circuit, subset)
                test_ideal_prob, test_noisy_prob, _ = self.evaluate_circuit(test_circuit)
                test_loss = test_ideal_prob - test_noisy_prob
                
                # 计算损失变化
                loss_change = target_loss - test_loss
                loss_improvement = loss_change > self.tolerance
                
                print(f"    测试结果:")
                print(f"      理想概率: {test_ideal_prob:.4f}")
                print(f"      噪声概率: {test_noisy_prob:.4f}")
                print(f"      当前损失: {test_loss:.4f}")
                print(f"      损失变化: {loss_change:+.4f} ({'减少' if loss_change > 0 else '增加'})")
                print(f"      基准损失: {target_loss:.4f}")
                
                # 如果排除这个子集后损失显著减少，说明这个子集包含问题
                if loss_improvement:
                    print(f"    ✓ 找到问题子集! 损失显著减少: {loss_change:.4f}")
                    print(f"    ✓ 更新候选集: {subset}")
                    candidates = subset
                    target_loss = test_loss
                    any_reduction = True
                    break
                else:
                    print(f"    ✗ 损失变化不显著，继续测试下一个子集")
            
            if not any_reduction:
                if n < len(candidates):
                    print(f"\n  所有子集测试完成，未找到显著改进")
                    print(f"  增加分割粒度: {n} → {min(n * 2, len(candidates))}")
                    n = min(n * 2, len(candidates))  # 增加分割粒度
                else:
                    print(f"\n  无法进一步缩小，算法结束")
                    break  # 无法进一步缩小
        
        print(f"\n{'='*50}")
        print(f"DDMin算法完成!")
        print(f"最终候选段: {candidates}")
        print(f"最终损失: {target_loss:.4f}")
        print(f"{'='*50}")
        
        return candidates
    
    def _split_into_subsets(self, items, n):
        """将列表分割为n个大致相等的子集"""
        if n >= len(items):
            return [[item] for item in items]
        
        chunk_size = len(items) // n
        remainder = len(items) % n
        
        subsets = []
        start = 0
        for i in range(n):
            end = start + chunk_size + (1 if i < remainder else 0)
            if start < len(items):
                subsets.append(items[start:end])
            start = end
        
        return [subset for subset in subsets if subset]  # 过滤空子集
    
    def analyze_problematic_segments(self, problematic_segments):
        """
        分析有问题的段
        
        Args:
            problematic_segments: 有问题的段索引列表
            
        Returns:
            dict: 分析结果
        """
        analysis = {
            'segment_count': len(problematic_segments),
            'segments': [],
            'operation_analysis': {},
            'qubit_analysis': {},
            'recommendations': []
        }
        
        all_operations = []
        all_qubits = set()
        
        for seg_idx in problematic_segments:
            if seg_idx < len(self.segments):
                segment = self.segments[seg_idx]
                seg_info = {
                    'layer_id': segment['layer_id'],
                    'description': segment['description'],
                    'instructions': len(segment['instructions']),
                    'operations': []
                }
                
                for inst in segment['instructions']:
                    op_info = {
                        'operation': inst['operation'],
                        'qubits': inst['qubits'],
                        'params': inst['params']
                    }
                    seg_info['operations'].append(op_info)
                    all_operations.append(inst['operation'])
                    all_qubits.update(inst['qubits'])
                
                analysis['segments'].append(seg_info)
        
        # 操作统计
        from collections import Counter
        op_counts = Counter(all_operations)
        analysis['operation_analysis'] = dict(op_counts)
        
        # 量子比特统计
        analysis['qubit_analysis']['affected_qubits'] = list(all_qubits)
        analysis['qubit_analysis']['qubit_count'] = len(all_qubits)
        
        # 生成建议
        if 'mcx' in op_counts or 'ccx' in op_counts:
            analysis['recommendations'].append("多控制门是主要错误源，考虑优化或替换")
        
        if len(all_qubits) > 6:
            analysis['recommendations'].append("涉及量子比特过多，考虑减少并行度")
        
        if 'x' in op_counts and op_counts['x'] > 10:
            analysis['recommendations'].append("X门过多可能导致累积误差")
        
        return analysis
    
    def debug_circuit(self, circuit, n_qubits, target_states):
        """
        主调试入口
        
        Args:
            circuit: 要调试的量子电路
            n_qubits: 量子比特数
            target_states: 目标状态列表
            
        Returns:
            dict: 调试结果
        """
        self.original_circuit = circuit
        self.target_states = target_states
        self.test_count = 0
        # 设定逻辑数据比特数量，用于测量与counts投影
        self.logical_n_qubits = n_qubits
        
        print(f"\n{'='*60}")
        print(f"量子电路Delta Debugging分析")
        print(f"{'='*60}")
        print(f"电路量子比特数: {n_qubits}")
        print(f"目标状态: {target_states}")
        print(f"容忍度: {self.tolerance}")
        
        # 提取电路段
        self.segments = self.extract_circuit_segments(circuit)
        print(f"电路分解为 {len(self.segments)} 个段:")
        for i, seg in enumerate(self.segments):
            print(f"  段{i}: {seg['description']}")
        
        # 运行DDMin
        problematic_segments = self.ddmin(self.segments)
        
        # 分析结果
        analysis = self.analyze_problematic_segments(problematic_segments)
        
        result = {
            'total_segments': len(self.segments),
            'problematic_segments': problematic_segments,
            'test_count': self.test_count,
            'analysis': analysis,
            'segments_info': self.segments
        }
        
        self._print_debug_report(result)
        return result
    
    def _print_debug_report(self, result):
        """打印调试报告"""
        print(f"\n{'='*60}")
        print(f"Delta Debugging 报告")
        print(f"{'='*60}")
        
        print(f"总测试次数: {result['test_count']}")
        print(f"总段数: {result['total_segments']}")
        print(f"有问题的段数: {len(result['problematic_segments'])}")
        
        if result['problematic_segments']:
            print(f"\n有问题的段:")
            for seg_idx in result['problematic_segments']:
                seg = result['segments_info'][seg_idx]
                print(f"  段{seg_idx}: {seg['description']}")
                for inst in seg['instructions']:
                    print(f"    - {inst['operation']} on qubits {inst['qubits']}")
            
            analysis = result['analysis']
            print(f"\n操作统计:")
            for op, count in analysis['operation_analysis'].items():
                print(f"  {op}: {count}次")
            
            print(f"\n涉及的量子比特: {analysis['qubit_analysis']['affected_qubits']}")
            
        else:
            print("未找到明显的问题段")
    
    def save_debug_report(self, result, filename=None):
        """保存调试报告到JSON文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"delta_debug_report_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n调试报告已保存到: {filename}")
        return filename

def run_delta_debug_on_grover(executor, n_qubits=6, marked_states=[63]):
    """
    对Grover算法运行Delta Debugging
    
    Args:
        executor: 量子执行器
        n_qubits: 量子比特数
        marked_states: 目标状态
    """
    from program.grover_algorithm import grover_algorithm
    
    def get_test_circuit(name, **kwargs):
        """Get quantum algorithm circuit for delta debugging"""
        if name == "grover":
            return grover_algorithm(**kwargs)
        else:
            raise ValueError(f"Only grover algorithm is supported for delta debugging")
    
    print(f"正在为Grover算法创建Delta Debugger...")
    
    # 创建Grover电路
    circuit = get_test_circuit("grover", n_qubits=n_qubits, marked_states=marked_states)
    
    # 创建调试器
    debugger = QuantumDeltaDebugger(executor, marked_states, tolerance=0.02)
    
    # 运行调试
    result = debugger.debug_circuit(circuit, n_qubits, marked_states)
    
    # 保存报告
    report_file = debugger.save_debug_report(result)
    
    return result, report_file

if __name__ == "__main__":
    print("Delta Debugging模块已加载")
    print("使用示例:")
    print("from delta_debug import run_delta_debug_on_grover")
    print("result, report = run_delta_debug_on_grover(executor, n_qubits=6, marked_states=[63])") 