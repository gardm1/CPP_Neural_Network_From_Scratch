#include "network.hpp"

#include <vector>

int main() {
    std::vector<std::vector<double>> d = {
        // Training Data
        {3,   1.5, 1,   1},
        {2,   1,   0,   1},
        {4,   1.5, 1,   1},
        {3,   1,   0,   1},
        {3.5, 0.5, 1,   1},
        {2,   0.5, 0,   1},
        {5.5, 1,   1,   1},
        {1,   1,   0,   1},

        // Test Data
        {4,   1.5, 1,   2},
        {1,   0.5, 0,   2},
        {3.5, 1.5, 1,   2},
        {2.5, 1,   0,   2},
        {3.6, 0.6, 1,   2},
        {2,   0.7, 0,   2},
        {5.4, 1.2, 1,   2},
        {1.3, 1,   0,   2}
    };

    /* We got 2 flower types, blueand red.
    *  Red (1) flowers tend to have greater lenght and width
    *  Compared to blue (0) flowers.
    */

    /*   o __    Flower Type / output
    *   / \  \   Weights & bias weight
    *  o   o  b  Input (width & lenght) + Bias
    */

    FlowerPrediction::Network net({ 2, 1 }, 0.2);
    net.setDataset(d);


    net.train(10000);
    net.test();

    return 0;
}

