#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct layer LAYER;
typedef struct neuron NEURON;
typedef struct network NETWORK;
typedef struct dataset DATASET;

/*----- Network -----*/
struct network
{
    LAYER** layers;
    int numLayers;
    int inputSize;
    float lr;
};

NETWORK* newNetwork(int*, int, float);
void freeNetwork(NETWORK*);
void printNetwork(NETWORK*);
float* feedNetwork(NETWORK*, float*);
void backProp(NETWORK*, float*);
float pd_mse(float, float);
float pd_sig(float);
void updateNetworkDelta(NETWORK*, float);
void updateNetwork(NETWORK*);
void train(NETWORK*, float**, float**, int, int, int);
void shuffle(float**, float**, int);
void swap(float**, float**);

/*------ Layer  -----*/
struct layer
{
    NEURON** neurons;
    int nNeurons;
    int inputSize;
    float* inputs;
    float* outputs;
};


LAYER* newLayer(int, int);
void freeLayer(LAYER*);
void printLayer(LAYER*);
void activateLayer(LAYER*);
void updateLayerDelta(LAYER*, float);
void updateLayer(LAYER*);

/*----- Neuron -----*/
struct neuron
{
    float activation;
    float delta;
    float deltaAvg;
    float deltaN;
    float bias;
    float* weights;
    LAYER* parentLayer;
};

NEURON* newNeuron(LAYER*);
void freeNeuron(NEURON*);
void printNeuron(NEURON*);
void activateNeuron(NEURON*);
void updateNeuronDelta(NEURON*, float);
void updateNeuron(NEURON*);

/*----- Dataset -----*/
struct dataset{
    float** input;
    float** output;
    int dataSize;
};

DATASET* loadDataset(char*, int, int, int);
void freeDataset(DATASET*);