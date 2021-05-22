//
// Created by Marek on 22/05/2021.
//

#include "MLP.h"

DATASET* loadDataset(char* fileLoc, int dataSize, int inputSize, int outputSize)
{
    DATASET* ret = (DATASET*)malloc(sizeof(DATASET));
    ret->input = (float **)malloc(sizeof(float *) * dataSize);
    ret->output = (float **)malloc(sizeof(float *) * dataSize);
    ret->dataSize = dataSize;

    FILE *fp;
    fp = fopen(fileLoc, "r");
    for (int d = 0; d < dataSize; d++)
    {
        float *input = (float*)malloc(sizeof(float) * inputSize);
        float *truth = (float*)malloc(sizeof(float) * outputSize);


        for (int i = 0; i < inputSize; i++)
        {
            fscanf(fp, "%f", &input[i]);
        }

        for (int o = 0; o < outputSize; o++)
        {
            fscanf(fp, "%f", &truth[o]);
        }

        ret->input[d] = input;
        ret->output[d] = truth;
    }

    return ret;
}

void freeDataset(DATASET* dataset)
{
    for (int d = 0; d < dataset->dataSize; d++)
    {
        free(dataset->input[d]);
        free(dataset->output[d]);
    }
    free(dataset->input);
    free(dataset->output);
    free(dataset);
}