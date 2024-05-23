# Adaptively Perturbed Mirror Descent for Learning in Games
## tl;dr
This paper proposes a novel variant of Mirror Descent that achieves last-iterate convergence.

## Installation
```bash
pip install -r requirements.txt
```

## Run Experiments
In order to investigate the performance of APMD in Three-Player Biased Rock-Paper-Scissors with full feedback, execute the following command:
```bash
# D_{psi}=KL G=KL
$ python main.py n_trials=10 T=100000 game=three_biased_rps algorithm=APMD algorithm.learning_rate=0.1 algorithm.perturbation_strength=0.1 algorithm.random_init=True algorithm.regularizer=entropy algorithm.perturbation_divergence=kl feedback=full algorithm.update_slingshot_freq=100
# D_{psi}=KL G=Reverse KL
$ python main.py n_trials=10 T=100000 game=three_biased_rps algorithm=APMD algorithm.learning_rate=0.1 algorithm.perturbation_strength=0.1 algorithm.random_init=True algorithm.regularizer=entropy algorithm.perturbation_divergence=reverse_kl feedback=full algorithm.update_slingshot_freq=100
# D_{psi}=Squared L2 G=Squared L2
$ python main.py n_trials=10 T=100000 game=three_biased_rps algorithm=APMD algorithm.learning_rate=0.1 algorithm.perturbation_strength=1.0 algorithm.random_init=True algorithm.regularizer=l2 algorithm.perturbation_divergence=l2 feedback=full algorithm.update_slingshot_freq=20
```

To evaluate APMD via an experiment in Three-Player Biased Rock-Paper-Scissors with noisy feedback, execute the following command:
```bash
# D_{psi}=KL G=KL
$ python main.py n_trials=10 T=100000 game=three_biased_rps algorithm=APMD algorithm.learning_rate=0.01 algorithm.perturbation_strength=0.1 algorithm.random_init=True algorithm.regularizer=entropy algorithm.perturbation_divergence=kl feedback=noisy algorithm.update_slingshot_freq=1000
# D_{psi}=KL G=Reverse KL
$ python main.py n_trials=10 T=100000 game=three_biased_rps algorithm=APMD algorithm.learning_rate=0.01 algorithm.perturbation_strength=0.1 algorithm.random_init=True algorithm.regularizer=entropy algorithm.perturbation_divergence=reverse_kl feedback=noisy algorithm.update_slingshot_freq=1000
# D_{psi}=Squared L2 G=Squared L2
$ python main.py n_trials=10 T=100000 game=three_biased_rps algorithm=APMD algorithm.learning_rate=0.01 algorithm.perturbation_strength=1.0 algorithm.random_init=True algorithm.regularizer=l2 algorithm.perturbation_divergence=l2 feedback=noisy algorithm.update_slingshot_freq=200
```

## Reference
Kenshi Abe, Kaito Ariu, Mitsuki Sakamoto, and Atsushi Iwasaki. Adaptively perturbed mirror descent for learning in games. In ICML, 2024


Bibtex:
```
@inproceedings{
abe2024adaptively,
  title={Adaptively Perturbed Mirror Descent for Learning in Games},
  author={Abe, Kenshi and Ariu, Kaito and Sakamoto, Mitsuki and Iwasaki, Atsushi},
  booktitle={ICML},
  year={2024}
}
```