#include <stdio.h>
int globalVariable = 42; // 全局变量定义并初始化
int main() {
    // 打印全局变量的地址
    printf("Address of globalVariable: %p\n", (void*)&globalVariable);
    return 0;
}
