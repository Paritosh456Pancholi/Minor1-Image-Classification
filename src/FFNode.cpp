#include "FFNode.hpp"
#include <algorithm>

#include <cmath>
#include <cstdio>
#include <random>

FFNode::FFNode(Model& model,
               std::string name,
               Activation activation,
               uint16_t output_size,
               uint16_t input_size)
    : Node{model, std::move(name)}
    , activation_{activation}
    , output_size_{output_size}
    , input_size_{input_size}
{
    std::printf("%s: %d -> %d\n", name_.c_str(), input_size_, output_size_);

    weights_.resize(output_size_ * input_size_);

    biases_.resize(output_size_);

    activations_.resize(output_size_);

    activation_gradients_.resize(output_size_);
    weight_gradients_.resize(output_size_ * input_size_);
    bias_gradients_.resize(output_size_);
    input_gradients_.resize(input_size_);
}

void FFNode::init(rne_t& rne)
{
    num_t sigma;
    switch (activation_)
    {
    case Activation::ReLU:

        sigma = std::sqrt(2.0 / static_cast<num_t>(input_size_));
        break;
    case Activation::Softmax:
    default:
        sigma = std::sqrt(1.0 / static_cast<num_t>(input_size_));
        break;
    }

    auto dist = std::normal_distribution<num_t>{0.0, sigma};

    for (num_t& w : weights_)
    {
        w = dist(rne);
    }

    for (num_t& b : biases_)
    {
        b = 0.01;
    }
}

void FFNode::forward(num_t* inputs)
{
    last_input_ = inputs;

    for (size_t i = 0; i != output_size_; ++i)
    {
        num_t z{0.0};

        size_t offset = i * input_size_;

        for (size_t j = 0; j != input_size_; ++j)
        {
            z += weights_[offset + j] * inputs[j];
        }

        z += biases_[i];

        switch (activation_)
        {
        case Activation::ReLU:
            activations_[i] = std::max(z, num_t{0.0});
            break;
        case Activation::Softmax:
        default:
            activations_[i] = std::exp(z);
            break;
        }
    }

    if (activation_ == Activation::Softmax)
    {
        num_t sum_exp_z{0.0};
        for (size_t i = 0; i != output_size_; ++i)
        {
            sum_exp_z += activations_[i];
        }
        num_t inv_sum_exp_z = num_t{1.0} / sum_exp_z;
        for (size_t i = 0; i != output_size_; ++i)
        {
            activations_[i] *= inv_sum_exp_z;
        }
    }

    for (Node* subsequent : subsequents_)
    {
        subsequent->forward(activations_.data());
    }
}

void FFNode::reverse(num_t* gradients)
{
    for (size_t i = 0; i != output_size_; ++i)
    {
        num_t activation_grad{0.0};
        switch (activation_)
        {
        case Activation::ReLU:

            if (activations_[i] > num_t{0.0})
            {
                activation_grad = num_t{1.0};
            }
            else
            {
                activation_grad = num_t{0.0};
            }

            activation_gradients_[i] = gradients[i] * activation_grad;
            break;
        case Activation::Softmax:
        default:

            for (size_t j = 0; j != output_size_; ++j)
            {
                if (i == j)
                {
                    activation_grad += activations_[i]
                                       * (num_t{1.0} - activations_[i])
                                       * gradients[j];
                }
                else
                {
                    activation_grad
                        += -activations_[i] * activations_[j] * gradients[j];
                }
            }

            activation_gradients_[i] = activation_grad;
            break;
        }
    }

    for (size_t i = 0; i != output_size_; ++i)
    {
        bias_gradients_[i] += activation_gradients_[i];
    }

    std::fill(input_gradients_.begin(), input_gradients_.end(), num_t{0.0});

    for (size_t i = 0; i != output_size_; ++i)
    {
        size_t offset = i * input_size_;
        for (size_t j = 0; j != input_size_; ++j)
        {
            input_gradients_[j]
                += weights_[offset + j] * activation_gradients_[i];
        }
    }

    for (size_t i = 0; i != input_size_; ++i)
    {
        for (size_t j = 0; j != output_size_; ++j)
        {
            weight_gradients_[j * input_size_ + i]
                += last_input_[i] * activation_gradients_[j];
        }
    }

    for (Node* node : antecedents_)
    {
        node->reverse(input_gradients_.data());
    }
}

num_t* FFNode::param(size_t index)
{
    if (index < weights_.size())
    {
        return &weights_[index];
    }
    return &biases_[index - weights_.size()];
}

num_t* FFNode::gradient(size_t index)
{
    if (index < weights_.size())
    {
        return &weight_gradients_[index];
    }
    return &bias_gradients_[index - weights_.size()];
}

void FFNode::print() const
{
    std::printf("%s\n", name_.c_str());

    std::printf("Weights (%d x %d)\n", output_size_, input_size_);
    for (size_t i = 0; i != output_size_; ++i)
    {
        size_t offset = i * input_size_;
        for (size_t j = 0; j != input_size_; ++j)
        {
            std::printf("\t[%zu]%f", offset + j, weights_[offset + j]);
        }
        std::printf("\n");
    }
    std::printf("Biases (%d x 1)\n", output_size_);
    for (size_t i = 0; i != output_size_; ++i)
    {
        std::printf("\t%f\n", biases_[i]);
    }
    std::printf("\n");
}
