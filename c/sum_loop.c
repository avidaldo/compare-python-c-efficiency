// Adds up i % 7 for i = 0, 1, ..., count - 1 and prints the total and the seconds the loop took.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    long long count = atoll(argv[1]);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    long long total = 0;
    for (long long i = 0; i < count; i++) {
        total += i % 7;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double seconds = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("%lld %f\n", total, seconds);
    return 0;
}
