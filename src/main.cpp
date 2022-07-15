#include "CCELossNode.hpp"
#include "FFNode.hpp"
#include "GDOptimizer.hpp"
#include "MNIST.hpp"
#include "Model.hpp"
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <filesystem>

static constexpr size_t batch_size = 80;

Model create_model(std::ifstream& images,
                   std::ifstream& labels,
                   MNIST** mnist,
                   CCELossNode** loss)
{
    Model model{"ff"};

    *mnist = &model.add_node<MNIST>(images, labels);

    FFNode& hidden = model.add_node<FFNode>("hidden", Activation::ReLU, 10, 784);

    FFNode& output
        = model.add_node<FFNode>("output", Activation::Softmax, 10, 10);

    *loss = &model.add_node<CCELossNode>("loss", 10, batch_size);
    (*loss)->set_target((*mnist)->label());

    model.create_edge(hidden, **mnist);
    model.create_edge(output, hidden);
    model.create_edge(**loss, output);
    return model;
}

void train(char* argv[])
{
    std::printf("Executing training routine\n");

    std::ifstream images{
        std::filesystem::path{argv[0]} / "train-images-idx3-ubyte",
        std::ios::binary};

    std::ifstream labels{
        std::filesystem::path{argv[0]} / "train-labels-idx1-ubyte",
        std::ios::binary};

    MNIST* mnist;
    CCELossNode* loss;
    Model model = create_model(images, labels, &mnist, &loss);

    model.init();

    GDOptimizer optimizer{num_t{0.3}};

    size_t i = 0;
    for (; i != 256; ++i)
    {
        loss->reset_score();

        for (size_t j = 0; j != batch_size; ++j)
        {
            mnist->forward();
            loss->reverse();
        }

        model.train(optimizer);
    }

    std::printf("Ran %zu batches (%zu samples each)\n", i, batch_size);

    loss->print();

    std::ofstream out{
        std::filesystem::current_path() / (model.name() + ".params"),
        std::ios::binary};
    model.save(out);
}

void evaluate(char* argv[])
{
    std::printf("Executing evaluation routine\n");

    std::ifstream images{
        std::filesystem::path{argv[0]} / "t10k-images-idx3-ubyte",
        std::ios::binary};

    std::ifstream labels{
        std::filesystem::path{argv[0]} / "t10k-labels-idx1-ubyte",
        std::ios::binary};

    MNIST* mnist;
    CCELossNode* loss;

    Model model = create_model(images, labels, &mnist, &loss);

    std::ifstream params_file{std::filesystem::path{argv[1]}, std::ios::binary};
    model.load(params_file);

    for (size_t i = 0; i != mnist->size(); ++i)
    {
        mnist->forward();
    }
    loss->print();
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::printf("Supported commands include:\ntrain\nevaluate\n");
        return 1;
    }

    if (strcmp(argv[1], "train") == 0)
    {
        train(argv + 2);
    }
    else if (strcmp(argv[1], "evaluate") == 0)
    {
        evaluate(argv + 2);
    }
    else
    {
        std::printf("Argument %s is an unrecognized directive.\n", argv[1]);
    }

    return 0;
}
