/* Functions for multivariate Gaussian distributions
 */
#ifndef MVG_H
#define MVG_H

#include <armadillo>
#include <random>

/* Log-density of an isotropic multivariate Gaussian distribution
 *
 * state      : State at which to evaluate the density
 * data       : matrix needed for compatability
 */
double ld_mvg_iso(const arma::vec &state,
                  const arma::mat &data);

/* Simulate from an isotropic multivariate Gaussian
 *
 * generator : RNG
 * state     : simulated state
 * data      : matrix needed for compatability with RegenDist
 */
int rmvg_iso(std::mt19937_64 &generator,
             arma::vec &state,
             const arma::mat &data);

#endif
