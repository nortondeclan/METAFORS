# METAFORS
Code for a reservoir-computing implementation of *Meta-learning for Tailored Forecasting using Related Time Series* (METAFORS) described in [*D. A. Norton, E. Ott, A. Pomerance, B. Hunt, and M. Girvan, “Tailored forecasting from short time series via meta-learning,” (2025), arXiv:2501.16325*](https://doi.org/10.48550/arXiv.2501.16325).

Built on the *rescompy* python module for reservoir computing, [*D. Canaday, D. Kalra, A. Wikner, D. A. Norton, B. Hunt, and A. Pomerance, “rescompy 1.0.0: Fundamental Methods for Reservoir Computing in
Python,” GitHub (2024)*](https://github.com/PotomacResearch/rescompy).

![METAFORS_Schematic](https://github.com/user-attachments/assets/7486cc9f-e93a-4d5c-b047-d5b0bd5dad92)

**A Schematic of the METAFORS Meta-learning Method.** **(A)** We train the forecaster separately on each of the available long training series, $L_i$ , to construct both a model representation of each corresponding dynamical system, and a cold-start vector at every time-step. **(B)** We then divide the long signals into short sub-signals, $s_{ij}$, that are the same length as the short new system signals and **(C)** train the signal mapper to map these short signals to cold-start vectors, $m_{ij}$, appropriate for their start times, and suitable model parameters, $\theta_i$, for the forecaster. **(D)** Given a short new system signal, $s_{new}$, the signal mapper learns a suitable cold-start vector, $m_{n0}$, and model parameters, $\theta_n$. To make a prediction, we initialize the forecaster with the learned cold-start vector and drive the forecaster with the short new system signal to mitigate errors in the learned cold-start vector. Finally, we evolve the forecaster autonomously with the learned parameters.
