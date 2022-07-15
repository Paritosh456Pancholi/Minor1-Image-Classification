#pragma once

#include "Model.hpp"

class GDOptimizer : public Optimizer
{
public:
    GDOptimizer(num_t eta);

    void train(Node& node) override;

private:
    num_t eta_;
};
