#include "CCELossNode.hpp"
#include <limits>

CCELossNode::CCELossNode(Model& model,
                         std::string name,
                         uint16_t input_size,
                         size_t batch_size)
    : Node{model, std::move(name)}
    , input_size_{input_size}
    , inv_batch_size_{num_t{1.0} / static_cast<num_t>(batch_size)}
{
    gradients_.resize(input_size_);
}

void CCELossNode::forward(num_t* data)
{
    num_t max{0.0};
    size_t max_index;

    loss_ = num_t{0.0};
    for (size_t i = 0; i != input_size_; ++i)
    {
        if (data[i] > max)
        {
            max_index = i;
            max       = data[i];
        }

        loss_ -= target_[i]
                 * std::log(

                     std::max(data[i], std::numeric_limits<num_t>::epsilon()));

        if (target_[i] != num_t{0.0})
        {
            active_ = i;
        }
    }

    if (max_index == active_)
    {
        ++correct_;
    }
    else
    {
        ++incorrect_;
    }

    cumulative_loss_ += loss_;

    last_input_ = data;
}

void CCELossNode::reverse(num_t* data)
{
    for (size_t i = 0; i != input_size_; ++i)
    {
        gradients_[i] = -inv_batch_size_ * target_[i] / last_input_[i];
    }

    for (Node* node : antecedents_)
    {
        node->reverse(gradients_.data());
    }
}

void CCELossNode::print() const
{
    std::printf("Avg Loss: %f\t%f%% correct\n", avg_loss(), accuracy() * 100.0);
}

num_t CCELossNode::accuracy() const
{
    return static_cast<num_t>(correct_)
           / static_cast<num_t>(correct_ + incorrect_);
}
num_t CCELossNode::avg_loss() const
{
    return cumulative_loss_ / static_cast<num_t>(correct_ + incorrect_);
}

void CCELossNode::reset_score()
{
    cumulative_loss_ = num_t{0.0};
    correct_         = 0;
    incorrect_       = 0;
}
