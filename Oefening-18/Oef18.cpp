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
 * Running multiplier=1 ...
 *   → average time = 6.056819 sec
 * Running multiplier=2 ...
 *   → average time = 7.078272 sec
 * Running multiplier=4 ...
 *   → average time = 8.916764 sec
 * Running multiplier=8 ...
 *   → average time = 7.876505 sec
 * Running multiplier=16 ...
 *   → average time = 6.667308 sec
 * Running multiplier=32 ...
 *   → average time = 2.505875 sec
 * Running multiplier=64 ...
 *   → average time = 0.489982 sec
 * Running multiplier=128 ...
 *   → average time = 0.489659 sec
 * Running multiplier=256 ...
 *   → average time = 0.480043 sec
 *
 * Wat opvalt is het effect van false sharing:
 * wanneer meerdere threads schrijven naar dezelfde cacheline,
 * ontstaat veel synchronisatie tussen caches, wat de prestaties sterk verlaagt.
 * Bij kleine multipliers (1–8) liggen de threads dicht bij elkaar in het geheugen,
 * waardoor de uitvoeringstijd hoog is. Bij grotere multipliers (32–256)
 * schrijft elke thread in een aparte cacheline, verdwijnt false sharing,
 * en daalt de uitvoeringstijd sterk.
 * Dit toont aan dat de fysieke plaatsing van data in het geheugen cruciaal is voor prestaties bij multi-threading.
 */