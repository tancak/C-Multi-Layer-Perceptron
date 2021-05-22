//
// Created by Marek on 21/05/2021.
//

#include "MLP.h"

NETWORK* newNetwork(int* layerSizes, int numHiddenLayers, float lr)
{
    NETWORK* ret;
    ret = (NETWORK*)malloc(sizeof(NETWORK));
    ret->layers = (LAYER**)malloc(sizeof(LAYER*) * (numHiddenLayers + 1));
    ret->numLayers = numHiddenLayers + 1;
    ret->inputSize = layerSizes[0];
    ret->lr = lr;

    for (int n = 1; n < numHiddenLayers + 2; n++)
    {
        LAYER* l = newLayer(layerSizes[n], layerSizes[n-1]);
        ret->layers[n-1] = l;
    }

    return ret;
}

void freeNetwork(NETWORK* network)
{
    for (int l = 0; l < network->numLayers; l++)
    {
        freeLayer(network->layers[l]);
    }
    free(network->layers);
    free(network);
}

void printNetwork(NETWORK* network)
{
    for (int l = 0; l < network->numLayers; l++)
    {
        printf("Layer: %d\n", l);
        printLayer(network->layers[l]);
    }
}

float* feedNetwork(NETWORK* network, float* food)
{
    memcpy(network->layers[0]->inputs, food, sizeof(float*) * network->inputSize);
    activateLayer(network->layers[0]);

    for (int l = 1; l < network->numLayers; l++)
    {
        memcpy(network->layers[l]->inputs, network->layers[l-1]->outputs, sizeof(float*) * network->inputSize);
        activateLayer(network->layers[l]);
    }

    return network->layers[network->numLayers-1]->outputs;
}

void backProp(NETWORK* network, float* true)
{
    LAYER* last = network->layers[network->numLayers-1];
    for (int n = 0; n < last->nNeurons; n++)
    {
        NEURON* neu = last->neurons[n];
        neu->delta = network->lr * (pd_mse(neu->activation, true[n])) * pd_sig(neu->activation);
    }

    for (int l = network->numLayers-2; l >= 0; l--)
    {
        LAYER* lay = network->layers[l];
        LAYER* layNext = network->layers[l+1];
        for (int n = 0; n < lay->nNeurons; n++)
        {
            NEURON* neu = lay->neurons[n];
            float error = 0;
            for (int m = 0; m < layNext->nNeurons; m++)
            {
                error += layNext->neurons[m]->weights[n] * layNext->neurons[m]->delta;
            }
            neu->delta = network->lr * error * (pd_sig(neu->activation));
        }
    }
}

float pd_mse(float pred, float true)
{
    return true - pred;
}

float pd_sig(float pred)
{
    return pred * (1 - pred);
}

void updateNetworkDelta(NETWORK* network, float dn)
{
    for (int l = 0; l < network->numLayers; l++)
    {
        updateLayerDelta(network->layers[l], dn);
    }
}

void updateNetwork(NETWORK* network)
{
    for (int l = 0; l < network->numLayers; l++)
    {
        updateLayer(network->layers[l]);
    }
}

void train(NETWORK* network, float** datasetInput, float** datasetTrue, int datasetSize, int batchSize, int epochs)
{
    for (int epoch = 1; epoch < epochs + 1; epoch++)
    {
        shuffle(datasetInput, datasetTrue, datasetSize);
        for (int lb = 0; lb < datasetSize; lb += batchSize)
        {
            int ub = lb + batchSize;
            if (ub > datasetSize)
            {
                ub = datasetSize;
            }

            for (int d = lb; d < ub; d++)
            {
                feedNetwork(network, datasetInput[d]);
                backProp(network, datasetTrue[d]);
                float dn = ub-lb;
                dn = dn/batchSize;
                updateNetworkDelta(network, dn);
            }
            updateNetwork(network);
        }
    }
}

void shuffle(float** datasetInput, float** datasetTrue, int datasetSize)
{
    srand(time(NULL));
    int i;
    for(i = datasetSize-1; i > 0; i--) {
        int j = rand() % (i+1);
        swap(&datasetInput[i], &datasetInput[j]);
        swap(&datasetTrue[i], &datasetTrue[j]);
    }
}

void swap(float** a, float** b)
{
    float* temp = *a;
    *a = *b;
    *b = temp;
}