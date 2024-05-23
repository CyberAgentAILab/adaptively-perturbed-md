import numpy as np


def l2(y):
    u = sorted(y, reverse=True)
    cumsum_u = np.cumsum(u)
    ids = np.arange(len(u))[u + 1 / (np.arange(len(y)) + 1) * (1 - cumsum_u) > 0]
    rho = np.max(ids)
    lamb = 1 / (rho + 1) * (1 - cumsum_u[rho])
    x = np.maximum(y + lamb, 0)
    return x


class MD(object):
    def __init__(self, num_actions, regularizer, learning_rate, **kwargs):
        self.num_actions = num_actions
        self.regularizer = regularizer
        self.cum_gradient = np.zeros(num_actions)
        self.gradient = np.zeros(num_actions)
        self.learning_rate = learning_rate
        if kwargs['random_init']:
            k = np.random.exponential(scale=1.0, size=self.num_actions)
            self.strategy = k / k.sum()
        else:
            self.strategy = np.ones(num_actions) / num_actions
        self.average_strategy = np.zeros(num_actions)
        self.n = 0

    def name(self):
        alg_name = self.__class__.__name__
        alg_name += '_lr{}'.format(self.learning_rate)
        alg_name += '_{}'.format(self.regularizer)
        return alg_name

    def _md(self, cum_gradient, gradient):
        learning_rate = self.learning_rate if isinstance(self.learning_rate, (float, int)) else eval(self.learning_rate[0])(self.n + 1, *(self.learning_rate[1:]))
        if self.regularizer == 'l2':
            self.strategy = l2(self.strategy + learning_rate * gradient)
            self.strategy /= np.sum(self.strategy)
        elif self.regularizer == 'entropy':
            self.strategy = self.strategy * np.exp(learning_rate * gradient)
            self.strategy /= np.sum(self.strategy)
        else:
            raise RuntimeError('Illegal regularizer')

    def _calc_gradient(self):
        return self.cum_gradient, self.gradient

    def calc_strategy(self):
        self._md(*self._calc_gradient())
        self.average_strategy = (self.n * self.average_strategy + self.strategy) / (self.n + 1)
        self.n += 1
        return self.strategy

    def add_gradient(self, gradient):
        self.cum_gradient += gradient
        self.gradient = gradient
