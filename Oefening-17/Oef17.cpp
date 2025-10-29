#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdio>

int main(int argc, char const *argv[])
{
    std::vector<int> numbers = {243, 3, 4, 51, 234, 455, 76, 2, 4326, 78, 643};
    int min_val = numbers[0];

#pragma omp parallel for reduction(min:min_val)
    for (int i = 0; i < (int)numbers.size(); ++i) {
        if (numbers[i] < min_val) {
            min_val = numbers[i];
        }
    }

    printf("Minimum = %d\n", min_val);
    return 0;
}

// OUTPUT: Minimum = 2
