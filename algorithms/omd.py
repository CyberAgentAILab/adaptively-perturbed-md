import numpy as np
from algorithms import MD


def l2(y):
    u = sorted(y, reverse=True)
    cumsum_u = np.cumsum(u)
    ids = np.arange(len(u))[u + 1 / (np.arange(len(y)) + 1) * (1 - cumsum_u) > 0]
    rho = np.max(ids)
    lamb = 1 / (rho + 1) * (1 - cumsum_u[rho])
    x = np.maximum(y + lamb, 0)
    return x


class OMD(MD):
    def __init__(self, num_actions, regularizer, learning_rate, **kwargs):
        super().__init__(num_actions, regularizer, learning_rate, **kwargs)
        self.strategy_hat = self.strategy.copy()

    def _calc_gradient(self):
        return self.cum_gradient, self.gradient

    def calc_strategy(self):
        learning_rate = self.learning_rate if isinstance(self.learning_rate, (float, int)) else eval(self.learning_rate[0])(self.n + 1, *(self.learning_rate[1:]))
        gradient = self._calc_gradient()[1]
        if self.regularizer == 'l2':
            self.strategy_hat = l2(self.strategy_hat + learning_rate * gradient)
            self.strategy_hat /= np.sum(self.strategy_hat)
            self.strategy = l2(self.strategy_hat + learning_rate * gradient)
            self.strategy /= np.sum(self.strategy)
        elif self.regularizer == 'entropy':
            self.strategy_hat = self.strategy_hat * np.exp(learning_rate * gradient)
            self.strategy_hat /= np.sum(self.strategy_hat)
            self.strategy = self.strategy_hat * np.exp(learning_rate * gradient)
            self.strategy /= np.sum(self.strategy)
        else:
            raise RuntimeError('Illegal regularizer')
        self.average_strategy = (self.n * self.average_strategy + self.strategy) / (self.n + 1)
        self.n += 1
        return self.strategy
