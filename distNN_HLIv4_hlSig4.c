#include "fann.h"

#include <stdio.h>



int main()
{

    fann_type *calc_out;
    /*set up network architecture*/
    const unsigned int nLayers = 6;
    const unsigned int nInParams = 11;
    const unsigned int nH1Params = 11;
    const unsigned int nH2Params = 11;
    const unsigned int nH3Params = 11;
    const unsigned int nH4Params = 2;
    const unsigned int nOutParams = 1;
    const float desired_error = (const float) 0.005;
    const unsigned int max_epochs = 2000;
//    const unsigned int max_epochs = 15;
    const unsigned int epochs_between_reports = 100;
    const float learning_rate = (const float) 0.2;

    /*create network*/
    printf("Creating network.\n");
    struct fann *net = fann_create_standard(nLayers,nInParams,nH1Params,nH2Params,nH3Params,nH4Params,nOutParams);

    /*load training data*/
    char* trainDataFile = "./trainingData/ivL25/fann_trainingSet_RadixHLIv4_ivL25_5interps_500k_150kSame_WLmc20k_RandInterps_1.txt";
    printf("Loading Training Data.\nTraining Datafile: %s", trainDataFile);
    struct fann_train_data *data = fann_read_train_from_file(trainDataFile);
    
    char* testDataFile = "./trainingData/ivL25/fann_trainingSet_RadixHLIv4_ivL25_5interps_500k_150kSame_WLmc20k_RandInterps_2.txt";
    printf("Loading Testing Data.\nTest Datafile: %s", testDataFile);
    struct fann_train_data *testData = fann_read_train_from_file(testDataFile);

    fann_set_activation_function_layer(net, FANN_ELLIOT,1);
//    fann_set_activation_steepness_hidden(net,0.5);
    fann_set_activation_steepness_layer(net,0.5,1);
    fann_set_activation_function_layer(net, FANN_ELLIOT,2);
    fann_set_activation_steepness_layer(net, 0.5, 2);
    fann_set_activation_function_layer(net, FANN_ELLIOT,3);
    fann_set_activation_steepness_layer(net, 0.5, 3);
    fann_set_activation_function_layer(net, FANN_ELLIOT,4);
    fann_set_activation_steepness_layer(net, 0.5, 4);
    fann_set_activation_function_output(net, FANN_ELLIOT);
    fann_set_activation_steepness_output(net,0.5);
    
    /*set training algorithm to rprop*/
//    fann_set_training_algorithm(net,FANN_TRAIN_BATCH);
//    fann_set_learning_rate(net,learning_rate);
    fann_set_training_algorithm(net,FANN_TRAIN_RPROP);

    /*set stoping critera*/
    fann_set_train_stop_function(net, FANN_STOPFUNC_MSE);
//    fann_set_train_stop_function(net, FANN_STOPFUNC_BIT);
//    fann_set_bit_fail_limit(net,0.05f);

    /*initialize weights*/
//    fann_init_weights(net,data);
    fann_randomize_weights(net,-0.25,0.25);

    printf("\nthe bit fail limit is: %f\n\n", fann_get_bit_fail_limit(net));

    /*train the network*/
    printf("Training the network\n");
    fann_train_on_data(net, data, max_epochs, epochs_between_reports, desired_error);


    /*test the network*/
    printf("Testing the network. %f\n\n", fann_test_data(net, testData));


    unsigned int i = 0;
    unsigned int total = 0;
    unsigned int correct = 0;
    unsigned int accurate = 0;

    for(i = 0; i < fann_length_train_data(testData); i++) {
        calc_out = fann_run(net, testData->input[i]);
//        printf("Net input: %f %f %f %f %f %f %f %f %f %f %f\n", data->input[i][0],data->input[i][1],data->input[i][2],data->input[i][3],data->input[i][4],data->input[i][5],data->input[i][6],data->input[i][7],data->input[i][8],data->input[i][9],data->input[i][10]);

        if(i%25000==0)
            printf("%.1f %f\n",testData->output[i][0],calc_out[0]);

        if(testData->output[i][0] == 1) {
            if(calc_out[0] > 0) {
                correct++;
                if(calc_out[0] > 0.85)
                    accurate++;
            }
        }

        if(testData->output[i][0] == 0) {
            if(calc_out[0] <= 0.5) {
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
    printf("\nLearning rate: %f\n", learning_rate);
    printf("Network Arch: HL1 - 11 Elliot, HL2 - 11 Elliot, HL3 - 11 Elliot, HL4 - 2 Elliot,  Out - 1 Elliot\n");
    printf("Training Algorithm: RPROP\n");
    printf("Random initial weights\n");
    printf("\nSaving the network\n");

    /*save net*/
    fann_save(net, "./trainedNets/fann_donutSwypeDistance_HLIv4_elliotDeepArch_ElliotOut_randWeights_2kIt_ivL25_5randInterps.net");
//    fann_save(net, "Test.net");
    
    fann_print_connections(net);

    printf("Cleaning up\n");
    fann_destroy_train(data);
    fann_destroy_train(testData);
    fann_destroy(net);

    return 0;
}

