#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using num_t = float;

using rne_t = std::mt19937;

enum class Activation
{
    ReLU,
    Softmax
};

class Model;

class Node
{
public:
    Node(Model& model, std::string name);

    virtual void init(rne_t& rne) = 0;

    virtual void forward(num_t* inputs) = 0;

    virtual void reverse(num_t* gradients) = 0;

    virtual size_t param_count() const noexcept
    {
        return 0;
    }

    virtual num_t* param(size_t index)
    {
        return nullptr;
    }

    virtual num_t* gradient(size_t index)
    {
        return nullptr;
    }

    [[nodiscard]] std::string const& name() const noexcept
    {
        return name_;
    }

    virtual void print() const = 0;

protected:
    friend class Model;

    Model& model_;
    std::string name_;
    std::vector<Node*> antecedents_;
    std::vector<Node*> subsequents_;
};

class Optimizer
{
public:
    virtual void train(Node& node) = 0;
};

class Model
{
public:
    Model(std::string name);

    template <typename Node_t, typename... T>
    Node_t& add_node(T&&... args)
    {
        nodes_.emplace_back(
            std::make_unique<Node_t>(*this, std::forward<T>(args)...));
        return reinterpret_cast<Node_t&>(*nodes_.back());
    }

    void create_edge(Node& dst, Node& src);

    rne_t::result_type init(rne_t::result_type seed = 0);

    void train(Optimizer& optimizer);

    [[nodiscard]] std::string const& name() const noexcept
    {
        return name_;
    }

    void print() const;

    void save(std::ofstream& out);
    void load(std::ifstream& in);

private:
    friend class Node;

    std::string name_;
    std::vector<std::unique_ptr<Node>> nodes_;
};
