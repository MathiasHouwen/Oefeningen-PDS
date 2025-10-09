#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <string>

using namespace std;

vector<float> readFloats(const string &fname)
{
    vector<float> data;
    ifstream f(fname, std::ios::binary);

    if (!f) {
        cerr << "Kan bestand niet openen!" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

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

// Feedback: Const itteratie over datavalues
vector<int> histogram(vector<float> &data, int N, float max, float min) {
    vector<int> hist(N, 0);
    for (float val : data) {
        int index = binIndex(data, N, val, max, min);
        hist[index]++;
    }
    return hist;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N;
    if(rank == 0){
        cout << "Aantal bins: ";
        cin >> N;
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<float> fullData;
    int numPerProc;
    float globalMin, globalMax;

    if(rank == 0){
        fullData = readFloats("histvalues.dat");
        int totalCount = fullData.size();

        if(totalCount % size != 0){
            printf("Number of floats (%d) is not divisible by number of processes (%d)!\n", totalCount, size);
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        numPerProc = totalCount / size;
        globalMin = minVal(fullData);
        globalMax = maxVal(fullData);
    }

    MPI_Bcast(&globalMin, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&globalMax, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numPerProc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<float> localData(numPerProc);
    MPI_Scatter(rank == 0 ? fullData.data() : nullptr, numPerProc, MPI_FLOAT,
                localData.data(), numPerProc, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    vector<int> localHist = histogram(localData, N, globalMin, globalMax);

    vector<int> globalHist(N, 0);
    MPI_Reduce(localHist.data(), globalHist.data(), N, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0){
        cout << "Global histogram:" << endl;
        for(int i=0; i<N; i++){
            cout << i << ": " << globalHist[i] << endl;
        }
        cout << "Min: " << globalMin << ", Max: " << globalMax << endl;
    }

    MPI_Finalize();
    return 0;
}
