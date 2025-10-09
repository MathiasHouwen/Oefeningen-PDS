// mpiexec -n 2 .\Oef9.exe
#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    // Initialiseer MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            printf("Mag maar 2, processen hebben :)");
            fflush(stdout); // make sure it actually appears
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    const size_t N = 1048576;
    auto* sendbuf = (float*) malloc(N * sizeof(float));
    auto* recvbuf = (float*) malloc(N * sizeof(float));

    if(!sendbuf || !recvbuf) {
        printf("Memory allocation failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize the send array
    for(size_t i = 0; i < N; i++) {
        sendbuf[i] = static_cast<float>(rank);
    }

    int other = 1 - rank;

    MPI_Sendrecv(
        sendbuf, N, MPI_FLOAT, other, 0, // send buffer
        recvbuf, N, MPI_FLOAT, other, 0, // receive buffer
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    // Print first element of received array
    printf("I am process %d and I have received b(0) = %.2f\n", rank, recvbuf[0]);
    fflush(stdout);

    // Free memory
    free(sendbuf);
    free(recvbuf);

    // Finaliseer MPI
    MPI_Finalize();

    return 0;
}