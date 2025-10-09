#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if(rank == 0) {
            std::cerr << "Dit programme heeft maar exact 2 processen." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int repetitions = 1000000; // 10^6

    if(rank == 0) {
        std::cout << "buffer_size (bytes)\tduration (us)" << std::endl;
    }

    for(int n = 0; n <= 10; n++) {
        int buffer_size = static_cast<int>(pow(2, n)); // 2^n bytes
        std::vector<uint8_t> buffer(buffer_size, 0);

        double start_time = 0.0, end_time = 0.0;

        if(rank == 0) {
            start_time = MPI_Wtime();
            for(int i = 0; i < repetitions; i++) {
                MPI_Send(buffer.data(), buffer_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer.data(), buffer_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            end_time = MPI_Wtime();
            double avg_round_trip_us = ((end_time - start_time) / repetitions) * 1e6;
            std::cout << buffer_size << "\t\t" << avg_round_trip_us << std::endl;
        } else if(rank == 1) {
            for(int i = 0; i < repetitions; i++) {
                MPI_Recv(buffer.data(), buffer_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer.data(), buffer_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return 0;
}

/*
nodes: 1, tasks: 2
buffer_size (bytes)     duration (us)
1               0.308379
2               0.308197
4               0.308506
8               0.3076
16              0.307963
32              0.367624
64              0.343676
128             0.688649
256             0.753413
512             0.880915
1024            0.973978

nodes: 2, tasks: 1
buffer_size (bytes)     duration (us)
1               2.27545
2               2.25395
4               2.26456
8               2.2648
16              2.27072
32              2.33449
64              2.52208
128             2.59209
256             3.44726
512             3.64841
1024            3.90857

De resultaten tonen het verschil tussen communicatie binnen één node en tussen twee nodes.
Bij één node zijn de round-trip tijden laag (0,3–1 μs) dankzij shared memory,
terwijl tussen node communicatie via het netwerk veel trager is (2,3–3,9 μs). V
oor kleine buffers wordt de tijd bepaald door latency, voor grotere buffers speelt bandbreedte een grotere rol.
*/