//
// Created by Marek on 21/05/2021.
//

#include "MLP.h"

NEURON* newNeuron (LAYER* parentLayer)
{
    NEURON *ret;

    ret = (NEURON *) malloc(sizeof(NEURON));
    ret->weights = (float *) calloc(parentLayer->inputSize, sizeof(float) * parentLayer->inputSize);
    ret->bias = (float)rand()/(float)(RAND_MAX/2)-1;
    ret->delta = 0.0;
    ret->deltaAvg = 0.0;
    ret->deltaN = 0;
    ret->activation = 0.0;
    ret->parentLayer = parentLayer;

    for (int w = 0; w < parentLayer->nNeurons; w++)
    {
        ret->weights[w] = (float)rand()/(float)(RAND_MAX/2)-1;
    }

    return ret;
}

void freeNeuron(NEURON* neuron)
{
    free(neuron->weights);
    free(neuron);
}

void printNeuron(NEURON* neuron){
    printf("Weights: ");
    for (int n = 0; n < neuron->parentLayer->inputSize; n++)
    {
        printf("%f ", neuron->weights[n]);
    }
    printf("\nBias: %f\n", neuron->bias);
}

void activateNeuron(NEURON* neuron)
{
    float preActivation = neuron->bias;
    for (int n = 0; n < neuron->parentLayer->inputSize; n++)
    {
        preActivation += neuron->weights[n] * neuron->parentLayer->inputs[n];
    }

    neuron->activation = 1 / (1 + exp(0-preActivation));
}

void updateNeuronDelta(NEURON* neuron, float dn)
{
    neuron->deltaAvg += neuron->delta;
    neuron->deltaN += dn;
    neuron->delta = 0;
}

void updateNeuron(NEURON* neuron)
{
    neuron->deltaAvg = neuron->deltaAvg / neuron->deltaN;
    for (int w = 0; w < neuron->parentLayer->inputSize; w++)
    {
        neuron->weights[w] += neuron->deltaAvg * neuron->parentLayer->inputs[w];
    }
    neuron->bias += neuron->deltaAvg;
    neuron->deltaAvg = 0;
    neuron->deltaN = 0;
}