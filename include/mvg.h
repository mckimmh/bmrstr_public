/* Functions for multivariate Gaussian distributions
 */
#ifndef MVG_H
#define MVG_H

#include <armadillo>
#include <random>

/* Normalised log-density of a zero-mean multivariate Gaussian
 *
 * state     : State at which to evaluate the density
 * precision : Precision matrix
 */
double mvg_ld_normed(const arma::vec &state,
                     const arma::mat &precision);

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
