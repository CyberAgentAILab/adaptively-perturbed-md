import numpy as np

from functools import reduce
from games.base_game import BaseGame


class PolyMatrixGame(BaseGame):
    def __init__(self, utilities):
        self.utilities = utilities
        self.eye = np.eye(len(self.utilities), dtype=int)
        self.arange = np.arange(len(self.utilities))

    def num_players(self):
        return len(self.utilities)

    def num_actions(self, player_id):
        return self.utilities[0].shape[player_id]

    def full_feedback(self, strategies):
        n_players = self.num_players()
        strategy_matrices = np.array([strategies[i].reshape(np.ones(n_players, dtype=int) + (self.num_actions(i) - 1) * self.eye[i]) for i in range(n_players)], dtype=object)
        return [(self.utilities[i] * reduce(np.multiply, strategy_matrices[self.arange[self.arange != i]])).sum(axis=tuple(self.arange[self.arange != i])) for i in range(n_players)]

    def nash_conv(self, strategies):
        q_values = self.full_feedback(strategies)
        n_conv = 0
        for i in range(self.num_players()):
            n_conv += max(q_values[i]) - q_values[i] @ strategies[i]
        return n_conv

    def noisy_feedback(self, strategies):
        feedback = self.full_feedback(strategies)
        noise = [np.random.normal(0, 0.1, len(feedback[i])) for i in range(len(feedback))]
        return [feedback[i] + noise[i] for i in range(len(feedback))]

    @staticmethod
    def calc_poly_matrix_utilities(n_players, n_actions, utilities):
        u = np.zeros((n_players,) + tuple(n_actions))
        cnt = 0
        for i in range(n_players):
            for j in range(i + 1, n_players):
                shape = np.ones(n_players, dtype=int)
                shape[i] += n_actions[i] - 1
                shape[j] += n_actions[j] - 1
                u[i] += utilities[cnt].reshape(shape)
                u[j] += -utilities[cnt].reshape(shape)
                cnt += 1
        return u


def three_biased_rps():
    n_players = 3
    n_actions = [3, 3, 3]
    utilities = np.array([
        [[0, -1/3, 1],
        [1/3, 0, -1/3],
        [-1, 1/3, 0]], # 1 vs 2
        [[0, -1/3, 1],
        [1/3, 0, -1/3],
        [-1, 1/3, 0]], # 1 vs 3
        [[0, -1/3, 1],
        [1/3, 0, -1/3],
        [-1, 1/3, 0]], # 2 vs 3
    ], dtype=float)
    u = PolyMatrixGame.calc_poly_matrix_utilities(n_players, n_actions, utilities)
    utilities = np.array([
        # player 1
        [
            [[0 + 0, 0 + -1/3, 0 + 1],            # RRR, RRP, RRS
            [-1/3 + 0, -1/3 + -1/3, -1/3 + 1],    # RPR, RPP, RPS 
            [1 + 0, 1 + -1/3, 1 + 1]],            # RSR, RSP, RSS

            [[1/3 + 1/3, 1/3 + 0, 1/3 + -1/3],    # PRR, PRP, PRS
            [0 + 1/3, 0 + 0, 0 + -1/3],           # PPR, PPP, PPS 
            [-1/3 + 1/3, -1/3 + 0, -1/3 + -1/3]], # PSR, PSP, PSS
            [[-1 + -1, -1 + 1/3, -1 + 0],         # SRR, SRP, SRS
            [1/3 + -1, 1/3 + 1/3, 1/3 + 0],       # SPR, SPP, SPS 
            [0 + -1, 0 + 1/3, 0 + 0]],            # SSR, SSP, SSS
        ],
        # player 2
        [
            [[0 + 0, 0 + -1/3, 0 + 1],            # RRR, RRP, RRS
            [1/3 + 1/3, 1/3 + 0, 1/3 + -1/3],     # RPR, RPP, RPS 
            [-1 + -1, -1 + 1/3, -1 + 0]],         # RSR, RSP, RSS
            [[-1/3 + 0, -1/3 + -1/3, -1/3 + 1],   # PRR, PRP, PRS
            [0 + 1/3, 0 + 0, 0 + -1/3],           # PPR, PPP, PPS 
            [1/3 + -1, 1/3 + 1/3, 1/3 + 0]],      # PSR, PSP, PSS
            [[1 + 0, 1 + -1/3, 1 + 1],            # SRR, SRP, SRS
            [-1/3 + 1/3, -1/3 + 0, -1/3 + -1/3],  # SPR, SPP, SPS 
            [0 + -1, 0 + 1/3, 0 + 0]],            # SSR, SSP, SSS
        ],
        # player 3
        [
            [[0 + 0, 1/3 + 1/3, -1 + -1],         # RRR, RRP, RRS
            [0+ -1/3, 1/3 + 0, -1 + 1/3],         # RPR, RPP, RPS 
            [0 + 1, 1/3 + -1/3, -1 + 0]],         # RSR, RSP, RSS
            [[-1/3 + 0, 0 + 1/3, 1/3 + -1],       # PRR, PRP, PRS
            [-1/3 + -1/3, 0 + 0, 1/3 + 1/3],      # PPR, PPP, PPS 
            [-1/3 + 1, 0 + -1/3, 1/3 + 0]],       # PSR, PSP, PSS

            [[1 + 0, -1/3 + 1/3, 0 + -1],         # SRR, SRP, SRS
            [1 + -1/3, -1/3 + 0, 0 + 1/3],        # SPR, SPP, SPS 
            [1 + 1, -1/3 + -1/3, 0 + 0]],         # SSR, SSP, SSS
        ],
    ], dtype=float)
    assert np.array_equal(u, utilities)
    return PolyMatrixGame(u)


def random_zero_sum_polymatrix(n_players=3, n_actions=10):
    utilities = np.zeros((np.math.factorial(n_players), n_actions, n_actions))
    for i in range(np.math.factorial(n_players)):
        utilities[i] = np.random.uniform(-1, 1, size=[n_actions, n_actions])
    u = PolyMatrixGame.calc_poly_matrix_utilities(n_players, [n_actions]*n_players, utilities)
    return PolyMatrixGame(u)