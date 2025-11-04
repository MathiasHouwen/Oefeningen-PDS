#include <iostream>
#include <omp.h>
#include <cstdio>

int main() {
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        printf("Hello World from thread %d of %d\n", thread_id, num_threads);
    }
    return 0;
}

// OUTPUT:
// Hello World from thread 2 of 12
// Hello World from thread 0 of 12
// Hello World from thread 5 of 12
// Hello World from thread 4 of 12
// Hello World from thread 1 of 12
// Hello World from thread 6 of 12
// Hello World from thread 7 of 12
// Hello World from thread 3 of 12
// Hello World from thread 10 of 12
// Hello World from thread 11 of 12
// Hello World from thread 8 of 12
// Hello World from thread 9 of 12