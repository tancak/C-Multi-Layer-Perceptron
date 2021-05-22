#include "MLP.h"

int main()
{
    srand(time(NULL));

    printf("Hello, World!\n");
    int ls[] = {2, 2, 1};
    NETWORK *network = newNetwork(ls, 1, 0.5f);
    printNetwork(network);
    printf("\n");

    DATASET *ds = loadDataset("/mnt/c/Users/Marek/CLionProjects/MLP/xor.dat", 4, 2, 1);
    train(network, ds->input, ds->output, 4, 1, 10000);
    printNetwork(network);
    printf("\n");

    for (int i = 0; i < 4; i++)
    {
        feedNetwork(network, ds->input[i]);
        printf("%f\n", network->layers[1]->outputs[0]);
    }

    freeNetwork(network);
    freeDataset(ds);

    return 0;
}