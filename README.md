# bandit
Some multi-armed bandit code. The code

- Reproduce some of the results of chapter 2 of the "Reinforcement Learning, An introduction" book, by Sutton and Barto.
- Reproduce some of the results of [1].
- Fit model using [1] of experimental data.

File description:

```
analysis.py -- Analysis of the models fitted for the experimental data.
bandit.py -- Reproduce results from Sutton's book.
contextual_bandit.py -- Simulation of the actual experiment.
filter.py -- Filter parameters for filtfilt (doesn't really belongs here).
ml.py -- Maximum Likelihood estimation of bandit parameters.
models.py -- Simulation of different environments-agents.
parse.py -- Parse experimental data.
utils.py -- Some useful functions.
vis.py -- Visualization of results.
```


[1] N. D. Daw, "Trial-by-trial data analysis using computational models,"
    Decision making, affect, and learning: Attention and performance XXIII,
    vol. 23, p. 1, 2011.

