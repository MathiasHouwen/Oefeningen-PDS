#include <iostream>
#include <cmath>
#include <omp.h>
#include "timer.h"

bool is_prime(size_t number)
{
    if (number < 2) return false;
    size_t to_check = std::sqrt(number) + 1;
    for (size_t i = 2; i < to_check; ++i)
    {
        if (number % i == 0)
            return false;
    }
    return true;
}

int main()
{
    const size_t start = 2;
    const size_t end   = 108;
    const int numThreads = 4;

    omp_set_num_threads(numThreads);

    // Scheduling types
    std::string schedules[] = {"static", "dynamic", "guided"};

    for (auto &sched : schedules)
    {
        int prime_count = 0;
        Timer timer;
        timer.start();

        if (sched == "static")
        {
#pragma omp parallel for schedule(static) reduction(+:prime_count)
            for (size_t i = start; i <= end; ++i)
                if (is_prime(i))
                    prime_count += 1;
        }
        else if (sched == "dynamic")
        {
#pragma omp parallel for schedule(dynamic) reduction(+:prime_count)
            for (size_t i = start; i <= end; ++i)
                if (is_prime(i))
                    prime_count += 1;
        }
        else if (sched == "guided")
        {
#pragma omp parallel for schedule(guided) reduction(+:prime_count)
            for (size_t i = start; i <= end; ++i)
                if (is_prime(i))
                    prime_count += 1;
        }

        timer.stop();
        std::cout << "Schedule: " << sched
                  << " | Primes: " << prime_count
                  << " | Time: " << timer.durationNanoSeconds() * 1e-9 << " sec\n";
    }

    return 0;
}

/*
 * Schedule: static | Primes: 28 | Time: 2.8e-06 sec
 * Schedule: dynamic | Primes: 28 | Time: 3e-06 sec
 * Schedule: guided | Primes: 28 | Time: 2.9e-06 sec
 */