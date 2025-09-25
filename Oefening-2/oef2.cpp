using namespace std;

#include "timer.h"

#define N 20000
#define test 5

int index(int i, int j) {
    return i * N + j;
}

int rowMajor(vector<int> &data, AutoAverageTimer &t) {
    int som = 0;
    t.start();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            som += data[index(i, j)];
        }
    }
    t.stop();
    return som;
}

int colMajor(vector<int> &data, AutoAverageTimer &t) {
    int som = 0;
    t.start();
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            som += data[index(i, j)];
        }
    }
    t.stop();
    return som;
}

int main(int argc, char *argv[]) {
    std::vector<int> data(N * N);

    AutoAverageTimer t1("rowMajor");
    AutoAverageTimer t2("colMajor");

    for (int i = 0; i < test; i++) {
        int rm = rowMajor(data, t1);
        int cm = colMajor(data, t2);

        std::cout << rm << " " << cm << std::endl;
    }

    t1.report();
    t2.report();
}
/*
 * Zonder de optimalisatie van de compiler zijn de tijden
 * #rowMajor 1.31644 +/- 0.155309 sec (5 measurements)
 * #colMajor 3.15483 +/- 0.0492018 sec (5 measurements)
 * Dit is logisch want bij colom major zijn meer cash' misses
 * -> Dus de rowmajor mogelijk om er over te lopen is het beste
 *
 * Met compiler optimalasitie
 * #rowMajor 0.122002 +/- 0.0143088 sec (5 measurements)
 * #colMajor 0.171279 +/- 0.0269572 sec (5 measurements)
 * Hier zijn de tijde gelijkaardig, dus we zien dat de compiler het
 * overlopen van de 2D array optimaliseerd. Ook optimalisseerd die nog ander
 * toepassingen daarom ligt de voor rowmajer de tijd ook lager
 */