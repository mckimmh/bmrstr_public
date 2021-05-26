/* Brownian Motion Restore simulation in multiple dimensions
 */
#ifndef BMRSTR_H
#define BMRSTR_H

#include "log_post.h"
#include "regen_dist.h"
#include <armadillo>
#include <fstream>
#include <random>
#include <string>
#include <vector>

class BMRestore
{
public:
    /* Constructor
     * posterior   : LogPost object, which should contain data, log-density,
     *               grad log-density, laplacian log-density and dimension.
     * regen_dist  : RegenDist object.
     * logC        : constant.
     * kappa_bar   : Upper bound on the regeneration rate.
     * ntours      : Number of tours to simulate.
     * output_rate : Rate at which to output the state of the process
     */
    BMRestore(LogPost posterior,
              RegenDist regen_dist,
              double logC,
              double kappa_bar,
              int ntours = 10000,
              double output_rate = 1.0);
    
    // Set the regeneration distribution
    void set_regen_dist(RegenDist regen_dist);
    
    // Set constant C
    void set_logC(const double logC);
    
    // Set the upper bound on the regeneration rate
    void set_kappa_bar(const double kappa_bar);
    
    // Set the number of tours to simulate
    void set_ntours(const int ntours);
    
    // Set the rate at which the state of the process is outputted
    void set_output_rate(const double output_rate);
    
    // Set seed
    // Should only be called once, before any random numbers are generated
    void set_seed(const unsigned int s);
    
    // Compute the partial regeneration rate at state
    double kappa_partial(const arma::vec &state);
    
    // Compute the regeneration rate at state
    double kappa(const arma::vec &state);
    
    /* Generate fixed number of tours of Restore process
     *
     * Counts the number of evaluations of the target log-density,
     * its gradient and Laplacian.
     * Once process has been generated, output states as well as their
     * corresponding times and tour number can be printed to the console
     * or a file using the functions below.
     */
    void gen_fixed_ntours();
    
    // Returns dimension by value
    int get_dimension();
    
    // Print output times to console
    void print_output_times();
    
    // Print output times to ofstream file called file_name
    // Precondition: file is closed
    void print_output_times(std::ofstream &file,
                            std::string file_name);
    
    // Print output states to console
    void print_output_states();
    
    // Print output states to ofstream file called file_name
    // Precondition: file is closed
    void print_output_states(std::ofstream &file,
                             std::string file_name);
    
    // Print output tour number to console
    void print_output_tour_number();
    
    // Print output tour number to ofstream file called file_name
    // Precondition: file is closed
    void print_output_tour_number(std::ofstream &file,
                                  std::string file_name);
    
    // Get the sum of the number of evaluations of U, gradU, lapU
    int get_nevals();
    
    // Return constant logC
    double get_logC();
    
    // Return the upper bound on the regeneration rate
    double get_kappa_bar();
    
private:
    // Random number generator
    std::mt19937_64 m_gen;
    
    // Log posterior object
    LogPost m_posterior;
    
    // Regeneration distribution object
    RegenDist m_regen_dist;
    
    // Dimension, number of tours to simulate, number of samples generated,
    // number of energy/grad-energy/laplacian-energy evaluations,
    // indicator of whether moments have been estimated
    int m_dimension, m_ntours, m_nevals, m_tour_current;
    
    // Constant C, upperbound on the regeneration rate, log upperbound
    // on the regeneration rate, output rate, current time, sum of weights.
    double m_logC, m_kappa_bar, m_log_kappa_bar, m_output_rate, m_t_current;
    
    // Record of which tour the process was in at each output time
    std::vector<int> m_tour_number;
    
    // Output times
    std::vector<double> m_t;
    
    // Output state
    std::vector< arma::vec > m_x;
    
    // Current state
    arma::vec m_x_current;
    
    // Simulate a Brownian Motion at time s+t, when its state at time s
    // is 'state'
    void bm(std::mt19937_64 &generator, arma::vec &state, double t);
    
    /* Simulate the state at the sooner of the next output time or the
     * next potential regeneration time
     *
     * minimal_regeneration : Indicator of whether to use the minimal
     *                        regeneration rate.
     * estimate_moments     : Indicates whether to not record output states and
     *                        instead estimate the first and second moments,
     *                        or to record all output states
     */
    void next_state();
};

#endif
