#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

using namespace std;

vector<float> readFloats(const string &fname)
{
    vector<float> data;
    ifstream f(fname, std::ios::binary);

    f.seekg(0, ios_base::end);
    int pos = f.tellg();
    f.seekg(0, ios_base::beg);
    if (pos <= 0)
        throw runtime_error("Can't seek in file " + fname + " or file has zero length");

    if (pos % sizeof(float) != 0)
        throw runtime_error("File " + fname + " doesn't contain an integer number of float32 values");

    int num = pos/sizeof(float);
    data.resize(num);

    f.read(reinterpret_cast<char*>(data.data()), pos);
    if (f.gcount() != pos)
        throw runtime_error("Incomplete read: " + to_string(f.gcount()) + " vs " + to_string(pos));
    return data;
}

float maxVal(vector<float> &data) {
    float max = data[0];
    for (float val: data) {
        if (val > max) {
            max = val;
        }
    }
    return max;
}

float minVal(vector<float> &data) {
    float min = data[0];
    for (float val: data) {
        if (val < min) {
            min = val;
        }
    }
    return min;
}

float bandwith(vector<float> &data, int N, float max, float min) {
    return (max - min) / N;
}

int binIndex(vector<float> &data, int N, float val, float max, float min) {
    float bandwidth = bandwith(data, N, max, min);
    if (val == max) {
        return N-1;
    }
    return (val - min) / bandwidth;
}

vector<int> histogram(vector<float> &data, int N, float max, float min) {
    vector<int> hist(N, 0);
    for (float val : data) {
        int index = binIndex(data, N, val, max, min);
        hist[index]++;
    }
    return hist;
}

int main(int argc, char *argv[]) {
    int N;
    cout << "Aantal bins:";
    cin >> N;

    vector<float> data = readFloats("histvalues.dat");
    cout << "maximale waarden: " << maxVal(data) << endl;
    cout << "minimale waarden: " << minVal(data) << endl;

    vector<int> hist = histogram(data, N, maxVal(data), minVal(data));
    for (int i = 0; i < N; i++) {
        cout << i << ": " << hist[i] << "\n";
    }
}
