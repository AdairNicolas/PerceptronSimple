#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_INPUTS 2
#define LEARNING_RATE 0.1
#define NUM_EPOCHS 100

float weights[NUM_INPUTS];
float bias;
int training_data[][NUM_INPUTS + 1] = {
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1}
};

float dot_product(float inputs[], float weights[], float bias) {
    float result = bias;
    for (int i = 0; i < NUM_INPUTS; i++) {
        result += inputs[i] * weights[i];
    }
    return result;
}

int activation(float input) {
    return input >= -0.000001 ? 1 : 0;
}

void train() {
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        for (int i = 0; i < 4; i++) {
            float inputs[NUM_INPUTS];
            for (int j = 0; j < NUM_INPUTS; j++) {
                inputs[j] = training_data[i][j];
            }
            int target = training_data[i][NUM_INPUTS];
            float prediction = activation(dot_product(inputs, weights, bias));
            float error = target - prediction;
            bias += LEARNING_RATE * error;
            for (int j = 0; j < NUM_INPUTS; j++) {
                weights[j] += LEARNING_RATE * error * inputs[j];
            }
        }
    }
}

void train_with_delta() {
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        printf("\nEpoch: %d\n",  epoch);
        for (int i = 0; i < 4; i++) {
            float inputs[NUM_INPUTS];
            for (int j = 0; j < NUM_INPUTS; j++) {
                inputs[j] = training_data[i][j];
            }
            int target = training_data[i][NUM_INPUTS];
            float result = dot_product(inputs, weights, bias);
            float prediction = activation(result);
            float error = target - prediction;
            bias += LEARNING_RATE * error;
            for (int j = 0; j < NUM_INPUTS; j++) {
                weights[j] += LEARNING_RATE * error * inputs[j];
            }
            
            printf("Weights: %f %f, Inputs: %f %f, Result: %f\n", 
            weights[0], weights[1], inputs[0], inputs[1], result);
            printf("Target: %d, Prediction: %f, Error: %f, Bias:%f\n", target, prediction, error, bias);
        }
    }
}

int main() {

    float inputs[][NUM_INPUTS] = {
      {0, 0},
      {0, 1},
      {1, 0},
      {1, 1}
    };

    srand(time(NULL));
    for (int i = 0; i < NUM_INPUTS; i++) {
        weights[i] = ((float)rand()/(float)(RAND_MAX)) * 2 - 1; // Initialize weights with random values between -1 and 1
    }
    bias = ((float)rand()/(float)(RAND_MAX)) * 2 - 1; // Initialize bias with random value between -1 and 1
    
    printf("\nAND pesos y bias aleatorio\n");
    printf("W1: %f, W2: %f, Bias: %f\n", weights[0], weights[1], bias);
    
    train_with_delta();
    
    printf("\nWeights: %f, %f, bias: %f\n", weights[0], weights[1], bias);
    for(int m = 0; m < 4;m++){
        int prediction = activation(dot_product(inputs[m], weights, bias));
        printf("Prediction for [%f, %f]: %d\n", inputs[m][0], inputs[m][1], prediction);
    }
    weights[0] = 1;
    weights[1] = 0;
    bias = -0.5;
    
    printf("\nAND pesos y bias definidos\n");
    printf("W1: %f, W2: %f, Bias: %f\n", weights[0], weights[1], bias);

    train_with_delta();

    printf("\nWeights: %f, %f, bias: %f\n", weights[0], weights[1], bias);
    for(int m = 0; m < 4;m++){
        int prediction = activation(dot_product(inputs[m], weights, bias));
        printf("Prediction for [%f, %f]: %d\n", inputs[m][0], inputs[m][1], prediction);
}

    return 0;
}

