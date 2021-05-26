/* Functions for multivariate Gaussian distributions
 */
#include "mvg.h"
#include <armadillo>
#include <cmath>
#include <random>

double mvg_ld_normed(const arma::vec &state,
                     const arma::mat &precision)
{
    int d = state.n_elem;
    arma::mat aux = state.t() * precision * state;
    return -0.5*d*log(2.0*M_PI) + 0.5*log(arma::det(precision)) - 0.5*aux(0,0);
}

int rmvg_iso(std::mt19937_64 &generator,
             arma::vec &state,
             const arma::mat &data)
{
    std::normal_distribution<double> rnorm(0.0, 1.0);
    for (arma::vec::iterator it = state.begin(); it != state.end(); ++it)
    {
        *it = rnorm(generator);
    }
    return 0;
}
