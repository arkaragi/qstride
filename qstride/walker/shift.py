"""
This module provides the ShiftOperator class, which allows for the creation and
application of shift operations in quantum walks.

The shift operator controls the movement of the quantum walker across the
positions (or nodes) in the quantum walk. The class supports different types of
shift operations that can be applied to a specified number of position qubits,
enabling the construction of multidimensional quantum walks.

Example usage
-------------
from qiskit import QuantumCircuit
from shift_operator import ShiftOperator

# Instantiate a ShiftOperator for a system with 2 position qubits and 2 coin qubits
shift_op = ShiftOperator(num_position_qubits=2, num_coin_qubits=2)

# Create the quantum circuit for the shift operation
shift_circuit = shift_op.create_shift_circuit()

# Integrate with a main quantum circuit
main_qc = QuantumCircuit(4)  # Assuming 2 position qubits and 2 coin qubits
main_qc.compose(shift_circuit, qubits=[0, 1, 2, 3], inplace=True)
main_qc.draw('mpl')
"""

import matplotlib.pyplot as plt
import qiskit
from qiskit import QuantumCircuit


class ShiftOperator:
    """
    A class for creating and applying shift operators in discrete-time quantum walks.

    The ShiftOperator class facilitates the creation of quantum circuits that
    conditionally shift the state of position qubits based on the states of
    coin qubits, simulating the movement of the walker in a quantum walk.

    Parameters
    ----------
    num_position_qubits: int
        The number of qubits representing the position of the walker.

    num_coin_qubits: Optional[int], default=None
        The number of qubits used as the quantum coin. If None or 0, the shift
        will be unconditional, applied directly to the position qubits.
    """

    def __init__(self,
                 num_position_qubits: int,
                 num_coin_qubits: int):
        self.num_position_qubits = num_position_qubits
        self.num_coin_qubits = num_coin_qubits

        # Validate the qubit numbers
        self._validate_num_qubits()

        # Initialize the Shift circuit
        self.shift_circuit = QuantumCircuit(self.num_position_qubits + self.num_coin_qubits,
                                            name="shift_register")

    def _validate_num_qubits(self):
        """
        Validates that the number of position and coin qubits are positive integers.

        Raises
        ------
        ValueError
            If either num_position_qubits or num_coin_qubits are not positive integers.
        """
        if not isinstance(self.num_position_qubits, int) or self.num_position_qubits <= 0:
            raise ValueError("Number of position qubits must be a positive integer.")
        if not isinstance(self.num_coin_qubits, int) or self.num_coin_qubits <= 0:
            raise ValueError("Number of coin qubits must be a positive integer.")

    def _apply_left_shift(self):
        """
        Apply a left shift operation.
        """
        for i in range(self.num_position_qubits):
            # Control qubit from coin qubits, target qubit from position qubits
            self.shift_circuit.cx(self.num_coin_qubits - 1, self.num_coin_qubits + i)

    def _apply_right_shift(self):
        """
        Apply a right shift operation.
        """
        for i in range(self.num_position_qubits):
            # Control qubit from coin qubits, target qubit from position qubits
            self.shift_circuit.cx(self.num_coin_qubits - 1, self.num_coin_qubits + i)
            self.shift_circuit.x(self.num_coin_qubits + i)

    def create_shift_circuit(self,
                             shift_type: str = "both"):
        """
        Public method to create the shift circuit based on the coin qubits
        states.

        The shift operation is applied conditionally depending on the state
        of the coin qubits, or unconditionally if no coin qubits are present.

        Parameters
        ----------
        shift_type: str, default='both'
            Type of shift to apply. Options are:
            - 'left': Apply only the left shift.
            - 'right': Apply only the right shift.
            - 'both': Apply both left and right shifts.

        Returns
        -------
        QuantumCircuit
            The quantum circuit with the applied shift operations.

        Raises
        ------
        ValueError
            If an invalid shift_type is provided.
        """
        if shift_type not in {"left", "right", "both"}:
            raise ValueError("Invalid shift_type. "
                             "Must be 'left', 'right', or 'both'.")

        if shift_type == "left":
            self._apply_left_shift()
        elif shift_type == "right":
            self._apply_right_shift()
        elif shift_type == "both":
            self._apply_left_shift()
            self._apply_right_shift()

        return self.shift_circuit

    def visualize_shift_circuit(self):
        """
        Visualize the quantum circuit for the shift operation.
        """
        self.shift_circuit.draw('mpl')
        plt.show()


def main():
    # Example usage of ShiftOperator
    shift_operator = ShiftOperator(num_position_qubits=2, num_coin_qubits=1)

    # Create and visualize the shift circuit
    shift_circuit = shift_operator.create_shift_circuit(shift_type="both")
    shift_operator.visualize_shift_circuit()


if __name__ == "__main__":
    main()
