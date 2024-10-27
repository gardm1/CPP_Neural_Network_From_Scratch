#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include <stdlib.h>
#include <vector>
#include <random>
#include <cmath>

namespace FlowerPrediction {
    class Network {
    public:

        Network(std::vector<int> topology, double eta);
        ~Network();

        void setDataset(std::vector<std::vector<double>> d);

        double sigmoid(double x);

        double sigmoid_d(double x);

        void train(int epochs);
        void test();

    private:
        std::vector<double> weights;

        std::vector<std::vector<double>> traningData;
        std::vector<std::vector<double>> testData;

        double eta;
    };
}

#endif // !__NETWORK_HPP

