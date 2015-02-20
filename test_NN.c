#include <stdio.h>

#include "fann.h"

int main()
{
    fann_type *calc_out;
    struct fann *net;
    printf("Creating network.\n");

    char * netFile = "./trainedNets/fann_donutSwypeDistance_HLIv4_elliotDeepArch_ElliotOut_randWeights_2kIt_ivL25_5randInterps.net";

#ifdef FIXEDFANN
    net = fann_create_from_file(netFile);
#else
    net = fann_create_from_file(netFile);
#endif

    if(!net) {
        printf("Error creating ann --- ABORTING.\n");
        return -1;
    }

    fann_print_connections(net);
    fann_print_parameters(net);

    printf("Loading Testing Data.\n");
    struct fann_train_data *testData = fann_read_train_from_file("./trainingData/ivL25/fann_trainingSet_RadixHLIv4_ivL25_5interps_1M_350kSame_WLmc20k_RandInterps_2+3.txt");
    
    /*Create output file*/
    FILE *opf;
    opf = fopen("./fann_HLIv4_elliotDeepArch_ElliotOut_randWeights_wkIt_ivL25_5randInterps_testingOutput.txt","w");

    /*test the network*/
    printf("Testing the network. %f\n\n", fann_test_data(net, testData));

    unsigned int i = 0;
    unsigned int total = 0;
    unsigned int correct = 0;
    unsigned int accurate = 0;

    for(i = 0; i < fann_length_train_data(testData); i++) {
        calc_out = fann_run(net, testData->input[i]);
        fprintf(opf,"%f %f\n", testData->output[i][0], calc_out[0]);

        //used for debugging neural networks
        //printf("Net input: %f %f %f %f %f %f %f %f %f %f %f\n", data->input[i][0],data->input[i][1],data->input[i][2],data->input[i][3],data->input[i][4],data->input[i][5],data->input[i][6],data->input[i][7],data->input[i][8],data->input[i][9],data->input[i][10]);
        if(i%10000 == 0)
            printf("Net output: %f intended output: %f\n", calc_out[0], testData->output[i][0]);

        if(testData->output[i][0] == 1) {
            if(calc_out[0] > 0.5) {
                correct++;
                if(calc_out[0] > 0.9)
                    accurate++;
            }
        }

        if(testData->output[i][0] == 0) {
            if(calc_out[0] <= 0.5) {
                correct++;
                if(calc_out[0] < 0.1)
                    accurate++;
            }
        }

        total++;
    }

    fclose(opf); 
    double eff = (double)correct/(double)total;
    double accEff = (double)accurate/(double)total;

    /*print out super important info about the net*/
    printf("Testing Efficiency = %f\n", eff);
    printf("High Accuracy Efficiency = %f\n", accEff);
    printf("Total number of tests: %d\n", total);

    printf("Cleaning up\n");
    fann_destroy_train(testData);
    fann_destroy(net);

    return 0;
}
