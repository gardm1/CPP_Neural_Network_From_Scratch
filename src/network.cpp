#include "network.hpp"

namespace FlowerPrediction {
    std::random_device rd;
    std::mt19937 mt(rd());

    Network::Network(std::vector<int> topology, double eta) {
        this->eta = eta;

        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        weights.push_back(dist(mt));  // w1
        weights.push_back(dist(mt));  // w2
        weights.push_back(dist(mt));  // b
    }


    Network::~Network() {}

    void Network::setDataset(std::vector<std::vector<double>> d) {
        // 1 & 2 is the dataset's classes

        for (int i = 0; i < d.size(); i++) {
            if (d[i].back() == 1) {
                this->traningData.push_back(d[i]);
            }
            else if (d[i].back() == 2) {
                this->testData.push_back(d[i]);
            }
        }
    }

    double Network::sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    double Network::sigmoid_d(double x) {
        return x * (1.0 - x);

    }

    void Network::train(int epochs) {
        std::uniform_real_distribution<double> dist(0, traningData.size());

        for (int i = 0; i < epochs; i++) {
            int random = dist(mt);
            std::vector<double> d = traningData[random];
            double target = d[2];

            /******* FEED FORWARD *******/

            // Weighted average (neurons value)
            double activate = (d[0] * weights[0]) + (d[1] * weights[1]) + weights[2];

            // Activated average (between 0-1 (sigmoid))
            double prediction = sigmoid(activate);

            /******* BACK PROPAGATION (UPDATING ERROR & WEIGHTS) *******/

            // Squared error / cost function (find the lowest point on the graph)
            double error = std::powf((prediction - target), 2.0);

            if (i % 1000 == 0) {
                printf("error= %.5f\n", error);
            }

            // d/dprediction [ (prediction - target)^2 ] = 2(prediction - target)
            double error_prediction_d = 2 * (prediction - target);

            // Derivative of our squeezed prediction
            double prediction_activate_d = sigmoid_d(prediction);

            double activate_w1_d = d[0];  // d/dw [ d[0] * weights[0] ] = d[0]
            double activate_w2_d = d[1];  // d/dw [ d[1] * weights[1] ] = d[1]
            double activate_b_d = 1;      // d/db [ 1 * weights[3] ] = 1;

            // Chain Rule
            double error_w1_d = error_prediction_d * prediction_activate_d * activate_w1_d;
            double error_w2_d = error_prediction_d * prediction_activate_d * activate_w2_d;
            double error_b_d = error_prediction_d * prediction_activate_d * activate_b_d;

            weights[0] = weights[0] - eta * error_w1_d;
            weights[1] = weights[1] - eta * error_w2_d;
            weights[2] = weights[2] - eta * error_b_d;

        }
    }

    void Network::test() {
        double numCorrect = 0.0;
        double count = 0.0;

        for (int i = 0; i < testData.size(); i++) {
            count++;

            std::vector<double> d = testData[i];

            double activate = (d[0] * weights[0]) + (d[1] * weights[1]) + weights[2];
            double prediction = sigmoid(activate);

            printf("[ %.1f, %.1f, %.f ]\nprediction= %.5f\n\n", d[0], d[1], d[2], prediction);

            if (prediction >= 0.5 && d[2] == 1 || prediction < 0.5 && d[2] == 0) {
                numCorrect++;
            }
        }

        printf("Test Performance: %.3f %%\n", (numCorrect / count) * 100.0);
    }
}

