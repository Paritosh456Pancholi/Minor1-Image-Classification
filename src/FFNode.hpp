#pragma once

#include "Model.hpp"

#include <cstdint>
#include <vector>

class FFNode : public Node
{
public:
    FFNode(Model& model,
           std::string name,
           Activation activation,
           uint16_t output_size,
           uint16_t input_size);

    void init(rne_t& rne) override;

    void forward(num_t* inputs) override;

    void reverse(num_t* gradients) override;

    size_t param_count() const noexcept override
    {
        return (input_size_ + 1) * output_size_;
    }

    num_t* param(size_t index);
    num_t* gradient(size_t index);

    void print() const override;

private:
    Activation activation_;
    uint16_t output_size_;
    uint16_t input_size_;

    std::vector<num_t> weights_;

    std::vector<num_t> biases_;

    std::vector<num_t> activations_;

    std::vector<num_t> activation_gradients_;

    std::vector<num_t> weight_gradients_;
    std::vector<num_t> bias_gradients_;
    std::vector<num_t> input_gradients_;

    num_t* last_input_;
};
