/* Brownian Motion Restore simulation in multiple dimensions
 */
#include "bmrstr.h"
#include "log_post.h"
#include "regen_dist.h"
#include <armadillo>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

BMRestore::BMRestore(LogPost posterior,
                     RegenDist regen_dist,
                     double logC,
                     double kappa_bar,
                     int ntours,
                     double output_rate)
{
    m_posterior = posterior;
    if (!m_posterior.is_log_dens_constructed()){
        std::cerr << "LogPost doesn't contain log_dens\n";
    }
    if (!m_posterior.is_grad_log_dens_constructed()){
        std::cerr << "LogPost doesn't contain grad_log_dens\n";
    }
    if (!m_posterior.is_laplacian_log_dens_constructed()){
        std::cerr << "LogPost doesn't contain laplacian_log_dens\n";
    }
    
    m_regen_dist = regen_dist;
    m_logC = logC;
    m_kappa_bar = kappa_bar;
    m_log_kappa_bar = log(kappa_bar);
    m_ntours = ntours;
    m_output_rate = output_rate;
    m_dimension = m_posterior.get_dimension();
    m_t_current = 0;
    m_tour_current = 0;
    m_nevals = 0;
    m_x_current.set_size(m_dimension);
}

void BMRestore::set_regen_dist(RegenDist regen_dist)
{
    m_regen_dist = regen_dist;
}

void BMRestore::set_logC(const double logC)
{
    m_logC = logC;
}

void BMRestore::set_kappa_bar(const double kappa_bar)
{
    m_kappa_bar = kappa_bar;
    m_log_kappa_bar = log(kappa_bar);
}

void BMRestore::set_ntours(const int ntours)
{
    m_ntours = ntours;
}

void BMRestore::set_output_rate(const double output_rate)
{
    m_output_rate = output_rate;
}

void BMRestore::set_seed(const unsigned int s)
{
    m_gen.seed(s);
}

double BMRestore::kappa_partial(const arma::vec &state)
{
    // Compute the gradient
    arma::vec grad(m_dimension);
    m_posterior.update_grad_U(state, grad);
    
    return 0.5 * (arma::dot(grad, grad) - m_posterior.laplacian_U(state));
}

double BMRestore::kappa(const arma::vec &state)
{
    return kappa_partial(state) +
           exp(m_logC + m_regen_dist.log_dens(state)
               - m_posterior.log_dens(state));
}

void BMRestore::gen_fixed_ntours()
{
    // Regenerate and track number of target evaluations.
    // .rmu should return the sum of the number of evaluations of U, gradU, LapU.
    m_nevals += m_regen_dist.rmu(m_gen, m_x_current);
    
    while (m_tour_current < m_ntours)
    {
        next_state();
    }
}

int BMRestore::get_dimension()
{
    return m_dimension;
}

void BMRestore::print_output_times()
{
    for (std::vector<double>::iterator it = m_t.begin();
         it != m_t.end(); ++it){
        std::cout << *it << '\n';
    }
}

void BMRestore::print_output_times(std::ofstream &file,
                                   std::string file_name)
{
    if (file.is_open()){
        std::cerr << "file should be closed\n";
    } else {
        file.open(file_name);
        for (std::vector<double>::iterator it = m_t.begin();
             it != m_t.end(); ++it){
            file << *it << '\n';
        }
        file.close();
    }
}

void BMRestore::print_output_states()
{
    std::vector<arma::vec>::iterator row;
    arma::vec::iterator col;
    for (row = m_x.begin(); row != m_x.end(); ++row){
        for (col = (*row).begin(); col != (*row).end(); ++col){
            std::cout << *col << ' ';
        }
        std::cout << '\n';
    }
}

void BMRestore::print_output_states(std::ofstream &file,
                                    std::string file_name)
{
    if (file.is_open()){
        std::cerr << "file should be closed\n";
    } else {
        file.open(file_name);
        std::vector<arma::vec>::iterator row;
        arma::vec::iterator col;
        for (row = m_x.begin(); row != m_x.end(); ++row){
            for (col = (*row).begin(); col != (*row).end(); ++col){
                file << *col << ' ';
            }
            file << '\n';
        }
        file.close();
    }
}

int BMRestore::get_nevals()
{
    return m_nevals;
}

void BMRestore::print_output_tour_number()
{
    for (std::vector<int>::iterator it = m_tour_number.begin();
         it != m_tour_number.end(); ++it){
        std::cout << *it << '\n';
    }
}

void BMRestore::print_output_tour_number(std::ofstream &file,
                                         std::string file_name)
{
    if (file.is_open()){
        std::cerr << "file should be closed\n";
    } else {
        file.open(file_name);
        for (std::vector<int>::iterator it = m_tour_number.begin();
             it != m_tour_number.end(); ++it){
            file << *it << '\n';
        }
        file.close();
    }
}

double BMRestore::get_logC()
{
    return m_logC;
}

double BMRestore::get_kappa_bar()
{
    return m_kappa_bar;
}

void BMRestore::bm(std::mt19937_64 &generator, arma::vec &state, double t)
{
    std::normal_distribution<double> normal(0.0, 1.0);
    double n;
    for (arma::vec::iterator x = state.begin(); x != state.end(); ++x)
    {
        n = normal(generator);
        (*x) += sqrt(t) * n;
    }
}

void BMRestore::next_state()
{
    // RNGs: uniform, dominating PP, exogeneous output PP
    std::uniform_real_distribution<double> runif(0.0, 1.0);
    std::exponential_distribution<double> exp_kappa_bar(m_kappa_bar);
    std::exponential_distribution<double> exp_output(m_output_rate);
    
    // Simulate whether a potential regeneration event occurs
    // before the next output event
    double t_next_potential_regen = exp_kappa_bar(m_gen);
    double t_next_output = exp_output(m_gen);
    
    if (t_next_potential_regen < t_next_output){
        // Simulate state at next potential regeneration time
        m_t_current += t_next_potential_regen;
        bm(m_gen, m_x_current, t_next_potential_regen);
        
        // Simulate whether regeneration occurs
        double u = runif(m_gen);
        double kx, log_kx;
        
        kx = kappa(m_x_current);
        log_kx = log(kx);
        m_nevals += 3; // evaluate U, gradU, lapU
        
        if (log(u) < (log_kx - m_log_kappa_bar)){
            m_nevals += m_regen_dist.rmu(m_gen, m_x_current);
            m_tour_current++;
        }
    } else {
        // Simulate the state at the next output time
        // and record current state, time and tour number
        m_t_current += t_next_output;
        bm(m_gen, m_x_current, t_next_output);
        
        m_x.push_back(m_x_current);
        m_tour_number.push_back(m_tour_current);
        m_t.push_back(m_t_current);
    }
}
