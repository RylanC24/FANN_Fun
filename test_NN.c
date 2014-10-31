#include <stdio.h>

#include "fann.h"

int main()
{
    fann_type *calc_out;
    int ret = 0;

    struct fann *net;

    printf("Creating network.\n");

//    char * netFile = "fann220_float_donutSwypeDistance_HLIv4_elliotDeepArch_TanhOut_randWeights_bezTest.net";
    char * netFile = "Test.net";
#ifdef FIXEDFANN
    net = fann_create_from_file(netFile);
//    net = fann_create_from_file("fann_float_donutSwypeDistance_HLIv4_elliotDeepArch_ElliotOut_randWeights_2kIt_v2.net");
#else
    net = fann_create_from_file(netFile);
//    net = fann_create_from_file("fann_float_donutSwypeDistance_HLIv4_elliotDeepArch_ElliotOut_randWeights_2kIt_v2.net");
#endif

    if(!net)
    {
        printf("Error creating ann --- ABORTING.\n");
        return -1;
    }

    fann_print_connections(net);
    fann_print_parameters(net);

    printf("Loading Testing Data.\n");
    struct fann_train_data *testData = fann_read_train_from_file("./trainingData/fannTrainSet_RTWP_HLIv4_TanhOut_200k_100k40k_newWL_SloppyBezierCurves_rc3.txt");

   /*test the network*/
    printf("Testing the network. %f\n\n", fann_test_data(net, testData));

//    struct fann_train_data *testData = fann_read_train_from_file("./trainingData/fannTrainSet_RTWP_HLIv4_TanhOut_100k_50k20k_newWL_BezierCurves_rc1.txt");

    unsigned int i = 0;
    unsigned int total = 0;
    unsigned int correct = 0;
    unsigned int unsure = 0;
    unsigned int accurate = 0;

//    for(i = 0; i < 500; i++)
    for(i = 0; i < fann_length_train_data(testData); i++)
    {
        calc_out = fann_run(net, testData->input[i]);
//        printf("Net input: %f %f %f %f %f %f %f %f %f %f %f\n", data->input[i][0],data->input[i][1],data->input[i][2],data->input[i][3],data->input[i][4],data->input[i][5],data->input[i][6],data->input[i][7],data->input[i][8],data->input[i][9],data->input[i][10]);
        if(i%10000 == 0)
            printf("Net output: %f intended output: %f\n", calc_out[0], testData->output[i][0]);

        if(testData->output[i][0] == 1)
        {
            if(calc_out[0] > 0.5)
            {
                correct++;
                if(calc_out[0] > 0.85)
                    accurate++;
            }
        }

        if(testData->output[i][0] == 0)
        {
            if(calc_out[0] <= 0.5)
            {
                correct++;
                if(calc_out[0] < 0.15)
                    accurate++;
            }
        }

        total++;
    }

 
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
