/* Class representing a posterior density*/

#ifndef LOG_POST_H
#define LOG_POST_H

#include <armadillo>
#include <fstream>

class LogPost
{
public:
    /* Default Constructor
     *
     * dimension: dimension of the posterior
     */
    LogPost(int dimension = 1);
    
    /* Constructor
     *
     * dimension          : dimension of the posterior
     * data               : data used to form the posterior
     * log_dens           : function evaluating the log-density
     * grad_log_dens      : function evaluating the gradient
     *                      (passed by reference) of the log-density
     * laplacian_log_dens : function evaluating the laplacian of the log-density
     */
    LogPost(int dimension,
            const arma::mat& data,
            double (*log_dens)(const arma::vec& state,
                               const arma::mat& data),
            void (*grad_log_dens)(const arma::vec& state,
                                  arma::vec& grad,
                                  const arma::mat& data),
            double (*laplacian_log_dens)(const arma::vec& state,
                                         const arma::mat& data));
    
    // Sets Data
    void set_data(const arma::mat& data);
    
    // Sets the log density
    void set_log_dens(double (*log_dens)(const arma::vec& state,
                                         const arma::mat& data));
    
    // Sets the gradient of the log density
    void set_grad_log_dens(void (*grad_log_dens)(const arma::vec& state,
                                                 arma::vec& grad,
                                                 const arma::mat& data));
    
    // Sets the Laplacian of the log density
    void set_laplacian_log_dens(double (*laplacian_log_dens)
                                (const arma::vec& state,
                                 const arma::mat& data));
    
    // Get the dimension
    int get_dimension();
    
    // Print data to console output
    void print_data();
    
    // Print data to file
    void print_data(std::ofstream &file);
    
    // Log density at state
    double log_dens(const arma::vec& state);
    
    // Energy at state
    double U(const arma::vec& state);
    
    // Update grad_log_dens, the gradient of the log density at state
    void update_grad_log_dens(const arma::vec& state,
                              arma::vec& grad);
    
    // Update grad_U, the gradient of the energy at state
    void update_grad_U(const arma::vec& state,
                       arma::vec& grad);
    
    // Laplacian of the log density at state
    double laplacian_log_dens(const arma::vec& state);
    
    // Laplacian of the energy at state
    double laplacian_U(const arma::vec& state);
    
    // Return indicator of whether the
    // log density / grad log density / Laplacian log density
    // has been constructed.
    int is_log_dens_constructed();
    int is_grad_log_dens_constructed();
    int is_laplacian_log_dens_constructed();
private:
    // Data
    arma::mat m_data;
    
    // Dimension, indicators of whether
    // data/ m_log_dens/m_grad_log_dens/m_laplacian_log_dens
    // has been constructed, indicator of whether to transform the density
    int m_dimension, m_log_dens_constructed, m_data_constructed,
        m_grad_log_dens_constructed, m_laplacian_log_dens_constructed;
    
    // Log density of the posterior
    double (*m_log_dens)(const arma::vec& state,
                         const arma::mat& data);
    
    // Gradient of the log density of the posterior
    void (*m_grad_log_dens)(const arma::vec& state,
                            arma::vec& grad,
                            const arma::mat& data);
    
    // Laplacian of the log density of the posterior
    double (*m_laplacian_log_dens)(const arma::vec& state,
                                   const arma::mat& data);
};

#endif
