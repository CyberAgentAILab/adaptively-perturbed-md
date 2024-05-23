import numpy as np

from algorithms import FTRL


class OFTRL(FTRL):
    def __init__(self, num_actions, regularizer, learning_rate, **kwargs):
        super().__init__(num_actions, regularizer, learning_rate, **kwargs)
        self.prediction_vec = np.zeros(num_actions)
        self.pre_prediction_vec = np.zeros(num_actions)

    def _calc_gradient(self):
        return self.cum_gradient + self.prediction_vec, self.gradient + self.prediction_vec - self.pre_prediction_vec

    def add_gradient(self, gradient):
        self.cum_gradient += gradient
        self.gradient = gradient
        self.pre_prediction_vec = self.prediction_vec.copy()
        self.prediction_vec = gradient
