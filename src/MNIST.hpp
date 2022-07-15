#pragma once

#include "Model.hpp"
#include <fstream>

class MNIST : public Node
{
public:
    constexpr static size_t DIM = 28 * 28;

    MNIST(Model& model, std::ifstream& images, std::ifstream& labels);

    void init(rne_t&) override
    {}

    void forward(num_t* data = nullptr) override;

    void reverse(num_t* data = nullptr) override
    {}

    void read_next();

    void print() const override;

    [[nodiscard]] size_t size() const noexcept
    {
        return image_count_;
    }

    [[nodiscard]] num_t const* data() const noexcept
    {
        return data_;
    }

    [[nodiscard]] num_t* data() noexcept
    {
        return data_;
    }

    [[nodiscard]] num_t* label() noexcept
    {
        return label_;
    }

    [[nodiscard]] num_t const* label() const noexcept
    {
        return label_;
    }

    void print_last();

private:
    std::ifstream& images_;
    std::ifstream& labels_;
    uint32_t image_count_;

    char buf_[DIM];

    num_t data_[DIM];

    num_t label_[10];
};
