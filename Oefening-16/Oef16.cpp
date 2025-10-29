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

// OUTPUT: Hello World from thread 0 of 1