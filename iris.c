#include "MLP.h"

int main()
{
    srand(time(NULL));

    printf("Hello, World!\n");
    int ls[] = {4, 4, 3, 3};
    NETWORK *network = newNetwork(ls, 2, 0.5f);
    printNetwork(network);
    printf("\n");

    DATASET* ds = loadDataset("/mnt/c/Users/Marek/CLionProjects/MLP/iris.dat", 150, 4, 3);
    train(network, ds->input, ds->output, 150, 10, 100000);
    printNetwork(network);
    printf("\n");

    freeDataset(ds);
    ds = loadDataset("/mnt/c/Users/Marek/CLionProjects/MLP/iris.dat", 150, 4, 3);
    for (int i = 0; i < 5; i++)
    {
        feedNetwork(network, ds->input[i]);
        float* outputs = network->layers[2]->outputs;
        printf("%f, %f, %f\n", outputs[0], outputs[1], outputs[2]);
    }
    printf("\n");
    for (int i = 50; i < 55; i++)
    {
        feedNetwork(network, ds->input[i]);
        float* outputs = network->layers[2]->outputs;
        printf("%f, %f, %f\n", outputs[0], outputs[1], outputs[2]);
    }
    printf("\n");
    for (int i = 100; i < 105; i++)
    {
        feedNetwork(network, ds->input[i]);
        float* outputs = network->layers[2]->outputs;
        printf("%f, %f, %f\n", outputs[0], outputs[1], outputs[2]);
    }

    freeNetwork(network);
    freeDataset(ds);

    return 0;
}