/* Class Representing a Regeneration distribution
 */
#ifndef REGEN_DIST_H
#define REGEN_DIST_H

#include <armadillo>
#include <random>

class RegenDist
{
public:
    // Default constructor
    RegenDist(int dimension = 1);
    
    /* Constructor
     *
     * dimension : dimension of the state
     * data      : data to pass to member functions
     * log_dens  : evaluate the log density at state
     * rmu       : simulate once from the distribution, returns an integer
     *             representing some measure of computing cost
     */
    RegenDist(int dimension,
              const arma::mat &data,
              double (*log_dens)(const arma::vec &state,
                                 const arma::mat &data),
              int (*rmu)(std::mt19937_64 &generator,
                         arma::vec &state,
                         const arma::mat &data));
    
    // Change the data
    void set_data(const arma::mat &data);
    
    // Evaluate the log density at state
    double log_dens(const arma::vec &state);
    
    // Evaluate the energy at state
    double U(const arma::vec &state);
    
    // Simulate from the distribution
    // Stores the generated sample in 'state'
    int rmu(std::mt19937_64 &generator,
            arma::vec &state);
    
    // Get dimension
    int get_dimension();
    
    // Get the data
    void get_data(arma::mat &data);
    
private:
    // Data
    arma::mat m_data;
    
    // Dimension
    int m_dimension;
    
    // Log density
    double (*m_log_dens)(const arma::vec &state,
                         const arma::mat &data);
    
    // Simulate from the distribution
    int (*m_rmu)(std::mt19937_64 &generator,
                 arma::vec &state,
                 const arma::mat &data);
};

#endif
