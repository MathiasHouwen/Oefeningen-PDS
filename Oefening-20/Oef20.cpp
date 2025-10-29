#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <omp.h>
#include <string>
#include <cmath>

using namespace std;

vector<float> readFloats(const string &fname)
{
    vector<float> data;
    ifstream f(fname, ios::binary);

    f.seekg(0, ios_base::end);
    int pos = f.tellg();
    f.seekg(0, ios_base::beg);
    if (pos <= 0)
        throw runtime_error("Can't seek in file " + fname + " or file has zero length");

    if (pos % sizeof(float) != 0)
        throw runtime_error("File " + fname + " doesn't contain an integer number of float32 values");

    int num = pos / sizeof(float);
    data.resize(num);

    f.read(reinterpret_cast<char*>(data.data()), pos);
    if (f.gcount() != pos)
        throw runtime_error("Incomplete read: " + to_string(f.gcount()) + " vs " + to_string(pos));
    return data;
}

double maxVal(const std::vector<float> &data) {
    double max_val = data[0];

    #pragma omp parallel for reduction(max:max_val)
    for (size_t i = 0; i < data.size(); i++)
        max_val = std::max((double)data[i], max_val);

    return max_val;
}

double minVal(const std::vector<float> &data) {
    double min_val = data[0];

    #pragma omp parallel for reduction(min:min_val)
    for (size_t i = 0; i < data.size(); i++)
        min_val = std::min((double)data[i], min_val);

    return min_val;
}

float bandwith(float max, float min, int N) {
    return (max - min) / N;
}

int binIndex(float val, float min, float bandwidth, int N) {
    if (val == min + bandwidth*N) return N-1;
    return static_cast<int>((val - min) / bandwidth);
}

// Parallel histogram
vector<int> histogram(const vector<float> &data, int N, float max, float min) {
    vector<int> hist(N, 0);
    float bw = bandwith(max, min, N);

    int numThreads = omp_get_max_threads();
    vector<vector<int>> local_hist(numThreads, vector<int>(N, 0));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < data.size(); i++) {
            int idx = binIndex(data[i], min, bw, N);
            local_hist[tid][idx]++;
        }
    }

    for (int t = 0; t < numThreads; t++) {
        for (int i = 0; i < N; i++) {
            hist[i] += local_hist[t][i];
        }
    }

    return hist;
}

int main() {
    int N;
    cout << "Aantal bins: ";
    cin >> N;

    vector<float> data = readFloats("histvalues.dat");

    float max_value = maxVal(data);
    float min_value = minVal(data);

    cout << "Maximale waarde: " << max_value << endl;
    cout << "Minimale waarde: " << min_value << endl;

    vector<int> hist = histogram(data, N, max_value, min_value);

    for (int i = 0; i < N; i++) {
        cout << i << ": " << hist[i] << "\n";
    }

    return 0;
}

/*
 * Aantal bins:8
 * Maximale waarde: 171.849
 * Minimale waarde: 54.4014
 * 0: 96
 * 1: 5161
 * 2: 38534
 * 3: 46741
 * 4: 9216
 * 5: 7006
 * 6: 85634
 * 7: 7612
 */