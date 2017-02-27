# bandit
Some multi-armed bandit code. The code

- Reproduce some of the results of chapter 2 of the "Reinforcement Learning, An introduction" book, by Sutton and Barto.
- Reproduce some of the results of [1].
- Fit model using [1] of experimental data.

File descriptions:


* ```models.py``` -- Simulation of different environment-agent pairs. Each pair has a fixed structure.
* ```contextual_bandit.py``` -- Simulation of the contextual bandit experiment. The environment has the structure of the real experiment. It is possible to define different action rules.
* ```ml.py``` -- Maximum Likelihood estimation of bandit parameters.
* ```analysis.py``` -- Analysis of the models fitted to the experimental data.
* ```bandit.py``` -- Reproduce results from Sutton's book.
* ```parse.py``` -- Parse experimental data.
* ```vis.py``` -- Visualization of results.
* ```utils.py``` -- Some useful functions.
* ```filter.py``` -- Filter parameters for filtfilt (doesn't really belongs here).



[1] N. D. Daw, "Trial-by-trial data analysis using computational models,"
    Decision making, affect, and learning: Attention and performance XXIII,
    vol. 23, p. 1, 2011.

