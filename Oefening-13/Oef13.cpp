#include <iostream>
#include <math.h>
#include <thread>
#include <vector>

#include "../../../AppData/Local/Programs/CLion/bin/mingw/x86_64-w64-mingw32/include/math.h"
#define _USE_MATH_DEFINES

const int N = 100000000;   // aantal deelintervallen
const int NUM_THREADS = 4; // aantal threads

// f(x) = sqrt(1 - x^2)
double f(double x) {
    return std::sqrt(1.0 - x * x);
}

// thread-functie: berekent deelintegraal voor [start, end)
void integrate(int start, int end, double step, double& partial_sum) {
    double sum = 0.0;
    for (int i = start; i < end; ++i) {
        double x_i = -1.0 + i * step;
        sum += f(x_i) * step;
    }
    partial_sum = sum;
}

int main() {
    double step = 2.0 / N;              // breedte van elk deelsegment
    int chunk = N / NUM_THREADS;        // aantal stappen per thread
    std::vector<std::thread> threads;
    std::vector<double> results(NUM_THREADS, 0.0);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_THREADS; ++i) {
        int start = i * chunk;
        int end = (i == NUM_THREADS - 1) ? N : (i + 1) * chunk;
        threads.emplace_back(integrate, start, end, step, std::ref(results[i]));
    }

    for (auto& t : threads) {
        t.join();
    }

    double total_sum = 0.0;
    for (const double r : results) total_sum += r;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout.precision(10);
    std::cout << "Benadering van de integraal: " << total_sum << "\n";
    std::cout << "Verwachte waarde (pi/2):     " << M_PI / 2.0 << "\n";
    std::cout << "Fout: " << std::fabs(total_sum - M_PI / 2.0) << "\n";
    std::cout << "Uitvoeringstijd: " << elapsed.count() << " seconden\n";

    return 0;
}
