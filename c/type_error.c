#include <stdio.h>

double half(double number) {
    return number / 2;
}

int main(void) {
    printf("%f\n", half("ten"));
    return 0;
}
