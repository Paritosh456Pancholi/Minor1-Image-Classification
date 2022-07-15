#include "GDOptimizer.hpp"
#include "Model.hpp"
#include <cmath>

GDOptimizer::GDOptimizer(num_t eta)
    : eta_{eta}
{}

void GDOptimizer::train(Node& node)
{
    size_t param_count = node.param_count();
    for (size_t i = 0; i != param_count; ++i)
    {
        num_t& param    = *node.param(i);
        num_t& gradient = *node.gradient(i);

        param = param - eta_ * gradient;

        gradient = num_t{0.0};
    }
}
