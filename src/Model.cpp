#include "Model.hpp"

Node::Node(Model& model, std::string name)
    : model_(model)
    , name_{std::move(name)}
{}

Model::Model(std::string name)
    : name_{std::move(name)}
{}

void Model::create_edge(Node& dst, Node& src)
{
    dst.antecedents_.push_back(&src);
    src.subsequents_.push_back(&dst);
}

rne_t::result_type Model::init(rne_t::result_type seed)
{
    if (seed == 0)
    {
        std::random_device rd{};
        seed = rd();
    }
    std::printf("Initializing model parameters with seed: %u\n", seed);

    rne_t rne{seed};

    for (auto& node : nodes_)
    {
        node->init(rne);
    }

    return seed;
}

void Model::train(Optimizer& optimizer)
{
    for (auto&& node : nodes_)
    {
        optimizer.train(*node);
    }
}

void Model::print() const
{
    // Invoke "print" on each node in the order added
    for (auto&& node : nodes_)
    {
        node->print();
    }
}

void Model::save(std::ofstream& out)
{
    for (auto& node : nodes_)
    {
        size_t param_count = node->param_count();
        for (size_t i = 0; i != param_count; ++i)
        {
            out.write(
                reinterpret_cast<char const*>(node->param(i)), sizeof(num_t));
        }
    }
}

void Model::load(std::ifstream& in)
{
    for (auto& node : nodes_)
    {
        size_t param_count = node->param_count();
        for (size_t i = 0; i != param_count; ++i)
        {
            in.read(reinterpret_cast<char*>(node->param(i)), sizeof(num_t));
        }
    }
}
