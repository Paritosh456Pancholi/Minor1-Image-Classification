#pragma once

#include "Model.hpp"

class CCELossNode : public Node
{
public:
    CCELossNode(Model& model,
                std::string name,
                uint16_t input_size,
                size_t batch_size);

    void init(rne_t&) override
    {}

    void forward(num_t* inputs) override;

    void reverse(num_t* gradients = nullptr) override;

    void print() const override;

    void set_target(num_t const* target)
    {
        target_ = target;
    }

    num_t accuracy() const;
    num_t avg_loss() const;
    void reset_score();

private:
    uint16_t input_size_;

    num_t inv_batch_size_;
    num_t loss_;
    num_t const* target_;
    num_t* last_input_;

    size_t active_;
    num_t cumulative_loss_{0.0};

    size_t correct_   = 0;
    size_t incorrect_ = 0;
    std::vector<num_t> gradients_;
};
