/* Class representing a posterior density
 */

#include "log_post.h"
#include <armadillo>
#include <fstream>

LogPost::LogPost(int dimension)
{
    if (dimension < 1){
        std::cerr << "Dimension must be greater than or equal to 1\n";
    }
    m_dimension = dimension;
    m_data_constructed = 0;
    m_log_dens_constructed = 0;
    m_grad_log_dens_constructed = 0;
    m_laplacian_log_dens_constructed = 0;
}

LogPost::LogPost(int dimension,
                 const arma::mat& data,
                 double (*log_dens)(const arma::vec& state,
                                    const arma::mat& data),
                 void (*grad_log_dens)(const arma::vec& state,
                                       arma::vec& grad,
                                       const arma::mat& data),
                 double (*laplacian_log_dens)(const arma::vec& state,
                                              const arma::mat& data))
{
    if (dimension < 1){
        std::cerr << "Dimension must be greater than or equal to 1"
                  << std::endl;
    }
    m_dimension = dimension;
    m_data = data;
    m_log_dens = log_dens;
    m_grad_log_dens = grad_log_dens;
    m_laplacian_log_dens = laplacian_log_dens;
    m_data_constructed = 1;
    m_log_dens_constructed = 1;
    m_grad_log_dens_constructed = 1;
    m_laplacian_log_dens_constructed= 1;
}

void LogPost::set_data(const arma::mat& data)
{
    m_data = data;
    m_data_constructed = 1;
}

void LogPost::set_log_dens(double (*log_dens)(const arma::vec& state,
                                              const arma::mat& data))
{
    m_log_dens = log_dens;
    m_log_dens_constructed = 1;
}

void LogPost::set_grad_log_dens(void (*grad_log_dens)(const arma::vec& state,
                                                      arma::vec& grad,
                                                      const arma::mat& data))
{
    m_grad_log_dens = grad_log_dens;
    m_grad_log_dens_constructed = 1;
}

void LogPost::set_laplacian_log_dens(double (*laplacian_log_dens)
                                      (const arma::vec& state,
                                       const arma::mat& data))
{
    m_laplacian_log_dens = laplacian_log_dens;
    m_laplacian_log_dens_constructed = 1;
}

int LogPost::get_dimension()
{
    return m_dimension;
}

void LogPost::print_data()
{
    if (m_data_constructed){
        m_data.print();
    } else {
        std::cerr << "Data hasn't been constructed yet\n";
    }
}

void LogPost::print_data(std::ofstream &file)
{
    if (m_data_constructed)
    {
        m_data.print(file);
    } else {
        std::cerr << "Data hasn't been constructed yet\n";
    }
}

double LogPost::log_dens(const arma::vec& state)
{
    return m_log_dens(state, m_data);
}

double LogPost::U(const arma::vec& state)
{
    return -log_dens(state);
}

void LogPost::update_grad_log_dens(const arma::vec& state,
                                   arma::vec& grad)
{
    m_grad_log_dens(state, grad, m_data);
}

void LogPost::update_grad_U(const arma::vec& state,
                            arma::vec& grad)
{
    update_grad_log_dens(state, grad);
    grad *= -1;
}

double LogPost::laplacian_log_dens(const arma::vec& state)
{
    return m_laplacian_log_dens(state, m_data);
}

double LogPost::laplacian_U(const arma::vec& state)
{
    return -laplacian_log_dens(state);
}

int LogPost::is_log_dens_constructed()
{
    return m_log_dens_constructed;
}

int LogPost::is_grad_log_dens_constructed()
{
    return m_grad_log_dens_constructed;
}

int LogPost::is_laplacian_log_dens_constructed()
{
    return m_laplacian_log_dens_constructed;
}
