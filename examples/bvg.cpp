/* Test class BMRestore
 *
 * Simulate from a bivariate Gaussian distribution with covariance matrix
 *      1.2, 0.4
 *      0.4, 0.8
 * using an isotropic Gaussian regeneration distribution.
 * Prints a short, detailed path to "bmrstr_mvg_x1.txt", "bmrstr_mvg_ts1.txt",
 * "bmrstr_mvg_tours1.txt" and a long path to "bmrstr_mvg_x2.txt".
 */

#include "bmrstr.h"
#include "log_post.h"
#include "mvg.h"
#include "regen_dist.h"
#include <armadillo>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#define LOGC 2.07
#define KAPPA_BAR 100.0
#define NTOURS1 100
#define OUTPUT_RATE1 1000
#define NTOURS2 100000
#define OUTPUT_RATE2 1.0

// Target log-density, gradient and laplacian
double ldtarg(const arma::vec &state, const arma::mat &precision);
void grad_ldtarg(const arma::vec &state,
                 arma::vec &grad,
                 const arma::mat &precision);
double lap_ldtarg(const arma::vec &state, const arma::mat &precision);

int main()
{
    // Dimension, constant C, kappa_bar
    int d = 2;
    
    // Target density
    // Covariance matrix
    arma::mat targ_cov({{1.2, 0.4},
                        {0.4, 0.8}});
    // Precision matrix
    arma::mat targ_prec = arma::inv_sympd(targ_cov);
    LogPost gauss(d, targ_prec, ldtarg, grad_ldtarg, lap_ldtarg);
    
    // Regeneration distribution has identity covariance matrix
    arma::mat redundant_mat(d, d, arma::fill::eye);
    RegenDist mu(d, redundant_mat, ld_mvg_iso, rmvg_iso);
    
    double logC = LOGC;
    double kappa_bar = KAPPA_BAR;
    int ntours = NTOURS1;
    double output_rate = OUTPUT_RATE1;

    // Short detailed run
    BMRestore X1(gauss, mu, logC, kappa_bar, ntours, output_rate);
    
    // Generate samples
    X1.gen_fixed_ntours();
    
    // Print to a file
    std::ofstream file;
    X1.print_output_states(file, "bmrstr_x1.txt");
    X1.print_output_times(file, "bmrstr_ts1.txt");
    X1.print_output_tour_number(file, "bmrstr_tours1.txt");
    
    // Long run
    ntours = NTOURS2;
    output_rate = OUTPUT_RATE2;
    
    BMRestore X2(gauss, mu, logC, kappa_bar, ntours, output_rate);
    
    X2.gen_fixed_ntours();
    
    // Print to file
    X2.print_output_states(file, "bmrstr_x2.txt");
    
    return 0;
}

double ldtarg(const arma::vec &state, const arma::mat &precision)
{
    arma::mat aux = state.t() * precision * state;
    return -0.5 * aux(0,0);
}

void grad_ldtarg(const arma::vec &state,
                 arma::vec &grad,
                 const arma::mat &precision)
{
    grad = -(precision * state);
}

double lap_ldtarg(const arma::vec &state, const arma::mat &precision)
{
    return -arma::trace(precision);
}
