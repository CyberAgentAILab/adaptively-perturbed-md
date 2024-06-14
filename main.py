import hydra

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


from algorithms import *
from games import *
from omegaconf import ListConfig
from runner import runner


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg):
    np.random.seed(cfg.seed)
    # initialize game
    game_params = dict(cfg.game[cfg.game.game_name])
    game = eval(cfg.game.game_name)(**game_params)
    # initialize players
    params = dict(cfg.algorithm)
    learning_alg = eval(cfg.algorithm.alg_names)
    players = [learning_alg(game.num_actions(i), **params) for i in range(game.num_players())]
    alg_name = players[0].name()
    game_param_str = "_".join([f"{key}{value}" for key, value in game_params.items()])
    save_path = 'log/{}/{}/{}'.format(cfg.feedback, cfg.game.game_name+'_'+game_param_str, alg_name)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    # run experiments
    print('==========Run experiment==========')
    max_workers = int(mp.cpu_count() - 1) if cfg.max_workers == -1 else cfg.max_workers
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        arguments = [[trial_id, cfg, save_path, np.random.randint(0, 2 ** 32)] for trial_id in range(cfg.n_trials)]
        pool.map(run_experiment, *tuple(zip(*arguments)))
    print('==========Finish experiment==========')


def run_experiment(trial_id, cfg, save_path, seed):
    print(f'==========Start trial {trial_id}==========')
    np.random.seed(seed)
    # initialize game
    game_params = dict(cfg.game[cfg.game.game_name])
    game = eval(cfg.game.game_name)(**game_params)
    # initialize players
    params = dict(cfg.algorithm)
    learning_alg = eval(cfg.algorithm.alg_names)
    players = [learning_alg(game.num_actions(i), **params) for i in range(game.num_players())]
    log = runner.run(game, cfg.T, cfg.feedback, players)
    # save log
    df = log.to_dataframe()
    df = df.set_index('t')
    print(f'==========Finish trial {trial_id}==========')
    df.to_csv(save_path + f'/results_{trial_id}.csv')
    return df


if __name__ == '__main__':
    main()
