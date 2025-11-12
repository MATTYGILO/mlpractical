import numpy as np
import os

seed = 22102017
rng = np.random.RandomState(seed)
PENALTY_VERBOSE = int(os.environ.get("PENALTY_VERBOSE", "0")) == 1


class L1Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L1 norm.
    """

    def __init__(self, coefficient):
        """Create a new L1 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = abs(coefficient)

    def __call__(self, parameter):
        """Calculate L1 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        # Absolute value of parameters
        parameter = np.abs(parameter)

        # Sum of absolute values
        sigma = np.sum(parameter)

        # Multiply by the coefficient
        sigma *= self.coefficient

        return sigma

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """

        # Multiply the coefficient by the sign of the parameter
        return self.coefficient * np.sign(parameter)

    def __repr__(self):
        return 'L1Penalty({0})'.format(self.coefficient)


class L2Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L2 norm.
    """

    def __init__(self, coefficient):
        """Create a new L2 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = abs(coefficient)

    def __call__(self, parameter):
        """Calculate L2 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """

        # Square of parameters
        parameter = np.square(parameter)

        # Sum of squares
        sigma = np.sum(parameter)

        # Multiply by the coefficient
        sigma *= self.coefficient

        # Multiply by half for correct gradient
        sigma *= 0.5

        return sigma

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """

        return self.coefficient * parameter

    def __repr__(self):
        return 'L2Penalty({0})'.format(self.coefficient)

class L1L2MixPenalty(object):
    """L1 & L2 mix penalty.
    """

    def __init__(self, l1_coefficient, l2_coefficient):
        """Create a new L1 & L2 mix penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert l1_coefficient > 0. and l2_coefficient > 0., 'Penalty coefficient must be positive.'

        # The ratio between L1 and L2 penalties
        self.l1_coefficient = abs(l1_coefficient)
        self.l2_coefficient = abs(l2_coefficient)

        # Define the two penalties
        self.l1_penalty = L1Penalty(l1_coefficient)
        self.l2_penalty = L2Penalty(l2_coefficient)


    def __call__(self, parameter):
        """Calculate L1 & L2 mix penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        return self.l1_penalty(parameter) + self.l2_penalty(parameter)

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        return self.l1_penalty.grad(parameter) + self.l2_penalty.grad(parameter)

    def __repr__(self):
        return 'L1L2MixPenalty({l1_coeff: {}, l2_coeff: {}})'.format(self.l1_coefficient, self.l2_coefficient)
