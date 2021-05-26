# Regeneration-Enriched Brownian Motion

A <em>Restore process</em> [(Wang et al.)](https://arxiv.org/abs/1910.05037) arises from enriching an underlying continuous-time Markov process with regenerations from some <em>rebirth distribution</em> at the rate of an inhomogenous Poisson process. For a given choice of underlying process and rebirth distribution, the rate can be chosen such that the invariant distribution of the Restore process corresponds to a target distribution of interest. This means the Restore process may be used for Monte Carlo, in which case it is called a <em>Restore sampler</em>.

This directory contains a minimal implementation of a Restore sampler, with underlying process a Brownian motion. Code is written in C++ and uses the Armadillo library. File `bvg.cpp` in subdirectory `examples` shows how to use code in `src` and `include` to simulate a Restore process. You can compile this example program by typing `make bvg.out` at the terminal, then run through script `bvg.R`, which produces the output below and some other plots.

Since the Restore process in this case is built on Brownian motion, it is impossible to store the entirety of the process. Instead, the state of the process at the time of an exogeneous Poisson process is used as output. The image below shows output states for a short run of the process, when the invariant distribution is a (non-isotropic) bivariate Gaussian. <em>Tours</em> of the process, segments between regeneration times, are shown in different colours.

![Two-dimensional traceplot of Restore process](https://github.com/mckimmh/bmrstr_public/blob/main/examples/traceplot2d.png)

The following two images show marginal traceplots of the process. Tours are connected by lines. Times at which the process appears to jump are regenerations.

![Traceplot X1](https://github.com/mckimmh/bmrstr_public/blob/main/examples/traceplotX1.png)

![Traceplot X2](https://github.com/mckimmh/bmrstr_public/blob/main/examples/traceplotX2.png)
