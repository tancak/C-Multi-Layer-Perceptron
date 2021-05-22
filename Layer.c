//
// Created by Marek on 21/05/2021.
//

#include "MLP.h"

LAYER* newLayer(int nNeurons, int inputSize)
{
    LAYER* ret;
    ret = (LAYER*)malloc(sizeof(LAYER));
    ret->neurons = (NEURON**)malloc(sizeof(NEURON*) * nNeurons);
    ret->inputs = (float*)calloc(nNeurons, sizeof(float) * nNeurons);
    ret->outputs = (float*)calloc(nNeurons, sizeof(float) * nNeurons);

    ret->inputSize = inputSize;
    ret->nNeurons = nNeurons;

    for (int n = 0; n < nNeurons; n++)
    {
        ret->neurons[n] = newNeuron(ret);
    }

    return ret;
}

void freeLayer(LAYER* layer)
{
    for (int n = 0; n < layer->nNeurons; n++)
    {
        freeNeuron(layer->neurons[n]);
    }

    free(layer->neurons);
    free(layer->inputs);
    free(layer->outputs);
    free(layer);
}

void printLayer(LAYER* layer)
{
    for (int n = 0; n < layer->nNeurons; n++)
    {
        printf("Neuron: %d\n", n);
        printNeuron(layer->neurons[n]);
    }
}

void activateLayer(LAYER* layer)
{
    for (int n = 0; n < layer->nNeurons; n++)
    {
        activateNeuron(layer->neurons[n]);
        layer->outputs[n] = layer->neurons[n]->activation;
    }
}

void updateLayerDelta(LAYER* layer, float dn)
{
    for (int n = 0; n < layer->nNeurons; n++)
    {
        updateNeuronDelta(layer->neurons[n], dn);
    }
}

void updateLayer(LAYER* layer)
{
    for (int n = 0; n < layer->nNeurons; n++)
    {
        updateNeuron(layer->neurons[n]);
    }
}