

from sandbox.rocky.tf.algos.bc import BC
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.lbfgs_optimizer import LbfgsOptimizer


class TRPO(BC):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            #optimizer = ConjugateGradientOptimizer(**optimizer_args)
            optimizer = None
        super(TRPO, self).__init__(optimizer=optimizer, **kwargs)
