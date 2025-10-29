#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>

#include "../../../AppData/Local/Programs/CLion/bin/mingw/x86_64-w64-mingw32/include/math.h"
#define _USE_MATH_DEFINES

using namespace std;

const int N = 100000000;   // aantal deelintervallen
const int NUM_THREADS = 4; // aantal threads

mutex mtx;            // voor veilige toegang tot de gedeelde som
double total_sum = 0.0;    // gedeelde variabele

double f(double x) {
    return sqrt(1.0 - x * x);
}

// Elke thread berekent zijn deel van de integraal en voegt het toe aan total_sum
void integrate(int start, int end, double step) {
    double local_sum = 0.0;
    for (int i = start; i < end; ++i) {
        double x_i = -1.0 + i * step;
        local_sum += f(x_i) * step;
    }

    // Kritische sectie: enkel één thread tegelijk mag total_sum aanpassen
    lock_guard<mutex> lock(mtx);
    total_sum += local_sum;
}

int main() {

    double step = 2.0 / N;
    int chunk = N / NUM_THREADS;
    vector<thread> threads;

    auto start_time = chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_THREADS; ++i) {
        int begin = i * chunk;
        int end = (i == NUM_THREADS - 1) ? N : (i + 1) * chunk;
        threads.emplace_back(integrate, begin, end, step);
    }

    for (auto& t : threads) t.join();

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout.precision(10);
    cout << "Benadering van de integraal: " << total_sum << "\n";
    cout << "Verwachte waarde (pi/2):     " << M_PI / 2.0 << "\n";
    cout << "Fout: " << fabs(total_sum - M_PI / 2.0) << "\n";
    cout << "Uitvoeringstijd: " << elapsed.count() << " seconden\n";

    return 0;

    /*
     * (Gemiddelde tijd)
     * t(oef13) = 0.4023047
     * t(oef14) = 0.408935 seconden
     *
     * Het verschil is klein (~0.0066 s)
     * maar consistent met de verwachting:
     * mutex-locking introduceert een klein beetje overhead
     * omdat de threads even moeten wachten wanneer ze hun
     * resultaat aan de gedeelde som toevoegen.
     */
}
