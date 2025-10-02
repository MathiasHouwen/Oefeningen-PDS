#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // Initialiseer MPI
    MPI_Init(&argc, &argv);

    int rank, size;

    // Verkrijg de rank van dit proces
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Verkrijg het totaal aantal processen
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Print het bericht
    printf("Hello world, I am proc %d of total %d\n", rank, size);

    // Finaliseer MPI
    MPI_Finalize();

    return 0;
}
