"""
This module provides the QuantumWalk class, which implements quantum circuits
for quantum walks (QWalks).

The QuantumWalk class integrates coin and shift operations to simulate quantum
walks on various graph structures, with support for multiple steps.

Example usage
-------------
from qiskit import Aer
from quantum_walk import QuantumWalk

# Define a quantum walk with 2 position qubits and 2 coin qubits, using the Hadamard coin
qw = QuantumWalk(num_position_qubits=2, num_coin_qubits=2, coin_type='hadamard')

# Create the quantum walk circuit with 3 steps
qw.create_walk_circuit(num_steps=3)

# Visualize the full quantum walk circuit
qw.visualize_walk_circuit()

# Execute the quantum walk on a simulator
results = qw.execute_walk()
"""

from typing import Optional
import qiskit
from qiskit import Aer, transpile, execute
import matplotlib.pyplot as plt

from coin_operator import CoinOperator
from shift_operator import ShiftOperator


class QuantumWalk:
    """
    A class to create and execute quantum circuits for quantum walks.

    The QuantumWalk class combines coin and shift operations to perform quantum walks
    on a specified number of position qubits. It supports different types of coin operators
    (e.g., Hadamard, Grover) and allows for multiple steps in the quantum walk.

    Parameters
    ----------
    num_position_qubits: int
        The number of qubits used for the positions in the quantum walk.

    num_coin_qubits: int
        The number of qubits used for the coin in the quantum walk.

    coin_type: str, default="hadamard"
        The type of coin operator to apply. Options include 'hadamard', 'grover', and 'custom'.

    custom_coin_gate: qiskit.circuit.Gate, optional
        A custom quantum gate to use if the 'custom' coin_type is selected.
    """

    def __init__(self,
                 num_position_qubits: int,
                 num_coin_qubits: int,
                 coin_type: str = 'hadamard',
                 custom_coin_gate: Optional[qiskit.circuit.Gate] = None):
        self.num_position_qubits = num_position_qubits
        self.num_coin_qubits = num_coin_qubits
        self.coin_type = coin_type
        self.custom_coin_gate = custom_coin_gate

        # Initialize the coin and shift operators
        self.coin_operator = CoinOperator(num_coin_qubits=num_coin_qubits, coin_type=coin_type)
        self.shift_operator = ShiftOperator(num_position_qubits=num_position_qubits, num_coin_qubits=num_coin_qubits)

        # Placeholder for the full quantum walk circuit
        self.qw_circuit = None

    def create_walk_circuit(self, num_steps: int):
        """
        Create the full quantum walk circuit with the specified number of steps.

        Parameters
        ----------
        num_steps: int
            The number of steps in the quantum walk.
        """
        # Initialize the full quantum walk circuit
        self.qw_circuit = qiskit.QuantumCircuit(self.num_position_qubits + self.num_coin_qubits)

        for step in range(num_steps):
            # Apply the coin operator
            coin_circuit = self.coin_operator.create_coin_circuit(self.custom_coin_gate)
            self.qw_circuit.compose(coin_circuit, inplace=True)

            # Apply the shift operator
            shift_circuit = self.shift_operator.create_shift_circuit()
            self.qw_circuit.compose(shift_circuit, inplace=True)

    def visualize_walk_circuit(self):
        """
        Visualize the full quantum walk circuit.
        """
        if self.qw_circuit is None:
            raise ValueError("Quantum walk circuit has not been created. Please call 'create_walk_circuit' first.")

        self.qw_circuit.draw('mpl')
        plt.show()

    def execute_walk(self, shots: int = 1024):
        """
        Execute the quantum walk circuit on a simulator.

        Parameters
        ----------
        shots: int, default=1024
            The number of shots (repeated executions) for the quantum circuit.

        Returns
        -------
        dict
            The result counts from the quantum walk execution.
        """
        if self.qw_circuit is None:
            raise ValueError("Quantum walk circuit has not been created. Please call 'create_walk_circuit' first.")

        # Use the Qiskit Aer simulator
        simulator = Aer.get_backend('qasm_simulator')

        # Transpile and execute the quantum circuit
        transpiled_circuit = transpile(self.qw_circuit, simulator)
        result = execute(transpiled_circuit, backend=simulator, shots=shots).result()

        # Get the result counts
        counts = result.get_counts()
        return counts

    def visualize_results(self, counts: dict):
        """
        Visualize the results of the quantum walk execution.

        Parameters
        ----------
        counts: dict
            The result counts from the quantum walk execution.
        """
        qiskit.visualization.plot_histogram(counts)
        plt.show()


def main():
    # Define a quantum walk with 2 position qubits and 2 coin qubits, using the Hadamard coin
    qw = QuantumWalk(num_position_qubits=2, num_coin_qubits=2, coin_type='hadamard')

    # Create the quantum walk circuit with 3 steps
    qw.create_walk_circuit(num_steps=3)

    # Visualize the full quantum walk circuit
    qw.visualize_walk_circuit()

    # Execute the quantum walk on a simulator
    results = qw.execute_walk()

    # Visualize the results
    qw.visualize_results(results)


if __name__ == "__main__":
    main()
