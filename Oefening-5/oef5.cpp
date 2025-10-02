#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include "timer.h"

struct Entry {
    Entry* next;
    uint64_t padding[127];
};

double run_test(std::vector<Entry>& entries) {
    const size_t steps = 20'000'000;
    Entry* e = &entries[0];
    Timer t;

    t.start();
    for (size_t i = 0; i < steps; ++i) {
        e = e->next;
    }
    t.stop();

    double total_ns = t.durationNanoSeconds();
    return total_ns / static_cast<double>(steps);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <num_entries>\n";
        return 1;
    }

    size_t N = std::stoull(argv[1]);
    std::vector<Entry> entries(N);

    for (size_t i = 0; i < N; ++i) {
        entries[i].next = &entries[(i + 1) % N];
    }

    double ns_per_element = run_test(entries);
    std::cout << N << " : " << ns_per_element << std::endl;

    return 0;
}

/*
 *
 *
 *
 *
 *
 *
 *
 */