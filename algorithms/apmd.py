import numpy as np

from algorithms import MD


class APMD(MD):
    def __init__(
        self,
        num_actions,
        regularizer,
        learning_rate,
        perturbation_strength,
        perturbation_divergence,
        update_slingshot_freq,
        **kwargs
    ):
        self.perturbation_strength = perturbation_strength
        self.slingshot_strategy = np.ones(num_actions) / num_actions
        self.perturbation_divergence = perturbation_divergence
        self.update_slingshot_freq = update_slingshot_freq
        super().__init__(num_actions, regularizer, learning_rate, **kwargs)

    def name(self):
        alg_name = self.__class__.__name__
        if self.update_slingshot_freq is not None:
            alg_name += "_tsig{}".format(self.update_slingshot_freq)
        alg_name += "_mu{}".format(self.perturbation_strength)
        alg_name += "_div{}".format(self.perturbation_divergence)
        alg_name += "_lr{}".format(self.learning_rate)
        alg_name += "_{}".format(self.regularizer)
        return alg_name

    def add_gradient(self, gradient):
        if self.perturbation_divergence == "reverse_kl":
            mutation = (
                self.perturbation_strength
                * (self.slingshot_strategy - self.strategy)
                / self.strategy
            )
        elif self.perturbation_divergence == "kl":
            mutation = -self.perturbation_strength * (
                np.log(self.strategy / self.slingshot_strategy) + 1
            )
        elif self.perturbation_divergence == "l2":
            mutation = -self.perturbation_strength * (
                self.strategy - self.slingshot_strategy
            )
        elif self.perturbation_divergence == "chi":
            mutation = (
                -self.perturbation_strength
                * 2
                * (self.strategy - self.slingshot_strategy)
                / self.slingshot_strategy
            )
        elif self.perturbation_divergence == "hellinger":
            mutation = -self.perturbation_strength * (
                1 - np.sqrt(self.slingshot_strategy / self.strategy)
            )
        elif self.perturbation_divergence == "js":
            mutation = -self.perturbation_strength * np.log(
                2 * self.strategy / (self.strategy + self.slingshot_strategy)
            )
        elif self.perturbation_divergence == "sym_kl":
            mutation = -self.perturbation_strength * (
                np.log(self.strategy / self.slingshot_strategy)
                + 1
                - self.slingshot_strategy / self.strategy
            )
        else:
            raise RuntimeError("Illegal mutation divergence")
        self.cum_gradient += gradient + mutation
        self.gradient = gradient + mutation
        if (
            self.update_slingshot_freq is not None
            and self.n % self.update_slingshot_freq == 0
        ):
            self.n = 0
            self.slingshot_strategy = self.strategy.copy()

    def add_bandit_gradient(self, utility, strategy, action):
        mutation = (
            self.perturbation_strength
            * (self.slingshot_strategy - self.strategy)
            / self.strategy
        )
        self.cum_gradient += utility / strategy[action] + mutation
        self.gradient = utility / strategy[action] + mutation
