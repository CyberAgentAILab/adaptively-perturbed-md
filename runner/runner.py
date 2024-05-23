from runner.logger import Logger


def run(game, T, feedback, players):
    n_players = game.num_players()

    log = Logger()
    for t in range(T):
        strategies = [player.strategy.copy() for player in players]
        if feedback == 'full':
            gradient = game.full_feedback(strategies)
            for i in range(n_players):
                players[i].add_gradient(gradient[i])
        elif feedback == 'noisy':
            gradient = game.noisy_feedback(strategies)
            for i in range(n_players):
                players[i].add_gradient(gradient[i])
        else:
            raise RuntimeError('Illegal feedback type {}'.format(feedback))
        for player in players:
            player.calc_strategy()

        log_interval = max(1, int(T / 10000))
        # record log
        if t < log_interval or t % log_interval  == 0:
            nash_conv_last_iterate = game.nash_conv([players[i].strategy for i in range(n_players)])
            log['t'].append(t)
            log['nash_conv_last_iterate'].append(nash_conv_last_iterate)

        if t % 100000 == 0:
            print('---Run Iteration {}---'.format(t))
            print('Nash conv of last-iterate strategies : {}'.format(nash_conv_last_iterate))
    return log
