#include <iostream>
#include <vector>
#include <cstdint>
#include <omp.h>
#include "timer.h"

const int64_t g_numLoops = 1 << 27; // 134 miljoen iteraties

void f(uint8_t *pBuffer, int offset)
{
    for (int64_t i = 0; i < g_numLoops; i++)
        pBuffer[offset] += 1;
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "Gebruik: " << argv[0] << " <multiplier>\n";
        return 1;
    }

    int multiplier = std::stoi(argv[1]);
    const int numThreads = 4;

    std::vector<uint8_t> buffer(multiplier * numThreads + 64, 0); // wat marge voor veiligheid

    Timer timer;
    timer.start();

#pragma omp parallel num_threads(numThreads)
    {
        int id = omp_get_thread_num();
        int offset = multiplier * id;
        f(buffer.data(), offset);
    }

    timer.stop();

    std::cout << "multiplier=" << multiplier
              << " tijd=" << timer.durationNanoSeconds() * 1e-9 << " sec"
              << std::endl;

    return 0;
}

/* OUTPUT
 * 1   multiplier=1 tijd=0.310429 sec
 * 2   multiplier=2 tijd=0.278223 sec
 * 4   multiplier=4 tijd=0.280607 sec
 * 8   multiplier=8 tijd=0.290469 sec
 * 16  multiplier=16 tijd=0.283338 sec
 * 32  multiplier=32 tijd=0.279948 sec
 * 64  multiplier=64 tijd=0.281985 sec
 * 128 multiplier=128 tijd=0.280005 sec
 * 256 multiplier=256 tijd=0.282316 sec
 *
 * Wanneer meerdere threads schrijven naar geheugenlocaties die binnen dezelfde cache line vallen,
 * ontstaat false sharing, wat de prestaties sterk verlaagt.
 */