import json
import datetime
import matplotlib.pyplot as plt
import numpy as np

class QuantumResultManager:
    """Quantum Result Management and Visualization Tools"""
    
    def __init__(self, base_filename=None):
        """
        Initialize result manager
        
        Args:
            base_filename: Base filename for output files
        """
        if base_filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"quantum_results_{timestamp}"
        
        self.base_filename = base_filename
        self.output_file = f"{base_filename}.json"
        
        # Initialize output data
        self.results_data = {
            "start_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "results": {}
        }
    
    def save_results_to_json(self, results, circuit_info, filename=None, backend_name="unknown"):
        """
        Save results to JSON file
        
        Args:
            results: Execution results (can be single result or dict of multiple results)
            circuit_info: Circuit information
            filename: Custom filename, if None uses default filename
            backend_name: Name of the backend used
            
        Returns:
            str: Path to saved file
        """
        # Prepare data
        data = {
            "start_time": self.results_data["start_time"],
            "end_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "backend": backend_name,
            "circuit_info": circuit_info,
            "results": results
        }
        
        # Use custom filename or default filename
        output_file = filename if filename else self.output_file
        
        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_file}")
        return output_file
    
    def generate_theoretical_analysis_chart(self, data, json_filename):
        """Generate theoretical analysis charts"""
        try:
            circuit_info = data['circuit_info']
            results = data['results']
            
            # Create 2x2 subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Subplot 1: Probability distribution comparison under different shots
            ax1 = axes[0, 0]
            shots_list = list(results.keys())
            
            # Select highest shots result to display probability distribution
            max_shots = max(shots_list)
            probs = results[max_shots]['theory']['probabilities']
            
            # Sort by probability
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            states = [item[0] for item in sorted_probs[:20]]  # Top 20 states
            probabilities = [item[1] for item in sorted_probs[:20]]
            
            ax1.bar(range(len(states)), probabilities)
            ax1.set_xlabel('Quantum States')
            ax1.set_ylabel('Probability')
            ax1.set_title(f'Top 20 States Probability Distribution ({max_shots} shots)')
            ax1.set_xticks(range(len(states)))
            ax1.set_xticklabels(states, rotation=45)
            
            # Subplot 2: Shots vs entropy
            ax2 = axes[0, 1]
            shots_values = []
            entropy_values = []
            
            for shots in shots_list:
                shots_values.append(shots)
                entropy_values.append(results[shots]['analysis']['entropy'])
            
            ax2.plot(shots_values, entropy_values, 'o-', linewidth=2, markersize=8)
            ax2.set_xlabel('Number of Shots')
            ax2.set_ylabel('Entropy')
            ax2.set_title('Entropy vs Shots')
            ax2.grid(True, alpha=0.3)
            
            # Subplot 3: Shots vs non-zero states
            ax3 = axes[1, 0]
            non_zero_states = [results[shots]['analysis']['non_zero_states'] for shots in shots_list]
            
            ax3.plot(shots_values, non_zero_states, 's-', linewidth=2, markersize=8, color='orange')
            ax3.set_xlabel('Number of Shots')
            ax3.set_ylabel('Non-zero States Count')
            ax3.set_title('Non-zero States vs Shots')
            ax3.grid(True, alpha=0.3)
            
            # Subplot 4: Circuit complexity information
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Display circuit information as text
            info_text = f"""
Circuit Information:
Algorithm: {circuit_info.get('algorithm', 'Unknown')}
Qubits: {circuit_info.get('num_qubits', 'Unknown')}
Depth: {circuit_info.get('depth', 'Unknown')}
Total Gates: {circuit_info.get('gate_count', 'Unknown')}

Test Configuration:
Shots tested: {', '.join(map(str, shots_list))}
Backend: {data.get('backend', 'Unknown')}
Test Type: {data.get('test_type', 'Unknown')}
            """
            
            ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=10,
                     verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            
            # Save chart
            chart_filename = json_filename.replace('.json', '_analysis.png')
            plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
            print(f"Analysis chart saved to: {chart_filename}")
            
            plt.show()
            
            return chart_filename
            
        except Exception as e:
            print(f"Error generating analysis chart: {e}")
            return None
    
    def generate_comparison_chart(self, theoretical_result, real_device_result, circuit_info, filename=None):
        """
        Generate comparison chart between theoretical and real device results
        
        Args:
            theoretical_result: Theoretical simulation results
            real_device_result: Real device execution results
            circuit_info: Circuit information
            filename: Custom filename for the chart
            
        Returns:
            str: Path to saved chart file
        """
        try:
            if not theoretical_result.get('success') or not real_device_result.get('success'):
                print("Cannot generate comparison chart: one or both results failed")
                return None
            
            # Get counts from both results
            theory_counts = theoretical_result.get('counts', {})
            real_counts = real_device_result.get('counts', {})
            
            # Get all unique states
            all_states = set(theory_counts.keys()) | set(real_counts.keys())
            all_states = sorted(all_states, key=lambda x: int(x, 2) if x.isdigit() or all(c in '01' for c in x) else 0)
            
            # Prepare data for plotting
            theory_probs = [theory_counts.get(state, 0) / theoretical_result.get('total_shots', 1) for state in all_states]
            real_probs = [real_counts.get(state, 0) / real_device_result.get('shots', 1) for state in all_states]
            
            # Create comparison chart
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Top plot: Probability comparison
            x = range(len(all_states))
            width = 0.35
            
            ax1.bar([i - width/2 for i in x], theory_probs, width, label='Theoretical', alpha=0.8)
            ax1.bar([i + width/2 for i in x], real_probs, width, label='Real Device', alpha=0.8)
            
            ax1.set_xlabel('Quantum States')
            ax1.set_ylabel('Probability')
            ax1.set_title('Theoretical vs Real Device Results Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Set x-axis labels (show every nth state to avoid crowding)
            n = max(1, len(all_states) // 20)
            ax1.set_xticks([i for i in x if i % n == 0])
            ax1.set_xticklabels([all_states[i] for i in x if i % n == 0], rotation=45)
            
            # Bottom plot: Difference
            differences = [real - theory for real, theory in zip(real_probs, theory_probs)]
            ax2.bar(x, differences, color='red', alpha=0.7)
            ax2.set_xlabel('Quantum States')
            ax2.set_ylabel('Probability Difference (Real - Theoretical)')
            ax2.set_title('Difference Between Real and Theoretical Results')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks([i for i in x if i % n == 0])
            ax2.set_xticklabels([all_states[i] for i in x if i % n == 0], rotation=45)
            
            # Add horizontal line at y=0
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            plt.tight_layout()
            
            # Save chart
            if filename is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"comparison_chart_{timestamp}.png"
            
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Comparison chart saved to: {filename}")
            
            plt.show()
            
            return filename
            
        except Exception as e:
            print(f"Error generating comparison chart: {e}")
            return None

def save_theoretical_test_results(circuit_name, circuit_params, all_results, circuit_info):
    """
    Save theoretical test results to JSON file
    
    Args:
        circuit_name: Name of the quantum algorithm
        circuit_params: Algorithm parameters
        all_results: Results for different shots
        circuit_info: Circuit information
        
    Returns:
        str: Path to saved file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"theoretical_test_{circuit_name}_{timestamp}.json"
    
    save_data = {
        "start_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "test_type": "theoretical_only",
        "circuit_info": circuit_info,
        "results": all_results
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Theoretical test results saved to: {filename}")
    return filename
