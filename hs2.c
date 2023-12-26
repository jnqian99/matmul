#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

typedef int64_t DATA_T;

#define BLOCK_SIZE 16

DATA_T max (DATA_T *a, int n, int i, int j, int k) {
    int m = i;
    if (j < n && a[j] > a[m]) {
        m = j;
    }
    if (k < n && a[k] > a[m]) {
        m = k;
    }
    return m;
}

void maxHeapify (DATA_T *a, int n, int i) {
    while (1) {
        int j = max(a, n, i, 2 * i + 1, 2 * i + 2);
        if (j == i) {
            break;
        }
        DATA_T t = a[i];
        a[i] = a[j];
        a[j] = t;
        i = j;
    }
}

void heapSort(DATA_T arr[], int n) {
    for (int i = n / 2 - 1; i >= 0; i--)
        maxHeapify(arr, n, i);

    for (int i = n - 1; i > 0; i--) {
        DATA_T temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;

        maxHeapify(arr, i, 0);
    }
}

void cacheBlockedHeapSort(DATA_T arr[], int n) {
    for (int i = n / 2 - 1; i >= 0; i--)
        maxHeapify(arr, n, i);

    for (int block = 0; block < n; block += BLOCK_SIZE) {
        int end = block + BLOCK_SIZE;
        if (end > n)
            end = n;

        for (int i = end - 1; i > block; i--) {
            DATA_T temp = arr[block];
            arr[block] = arr[i];
            arr[i] = temp;
            maxHeapify(arr, i, block);
        }

        for (int i = 0; i<(end-block)/2; i++) {
            DATA_T temp = arr[block+i];
            arr[block+i] = arr[end-1-i];
            arr[end-1-i] = temp;
        }
    }

    //printArray(arr, n);

    for (int i = n - 1; i > 0; i--) {
        DATA_T temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;

        maxHeapify(arr, i, 0);
    }
}

DATA_T* create_array(uint64_t length) {
    srand(time(NULL));
    DATA_T* array = (DATA_T*)malloc(length * sizeof(DATA_T));
    if (array == NULL) {
        return NULL;
    }
    for (uint64_t i = 0; i < length; i++) {
        array[i] = rand() % 1000;
    }
    return array;
}

void printArray(DATA_T arr[], int n) {
    for (int i = 0; i < n; i++)
        printf("%ld ", arr[i]);
    printf("\n");
}

int validateArray(const char* desc, DATA_T arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        if(arr[i]>arr[i+1]) {
            printf("%s failed! at value=%ld\n", desc, arr[i]);
            return 0;
        }
    }
    printf("%s validated OK!\n", desc);
    return 1;
}

int main() {
    int len=100;
    DATA_T* arr1 = create_array(len);
    DATA_T* arr2 = malloc(sizeof(DATA_T)*len);
    memcpy(arr2,arr1,sizeof(DATA_T)*len);

//    printf("Original array: ");
//    printArray(arr1, len);
//    printArray(arr2, len);

    printf("Heap-sorted array: ");
    heapSort(arr1, len);
    //printArray(arr1, len);
    validateArray("Heap Sort", arr1, len);

    printf("Cache-blocked heap-sorted array: ");
    cacheBlockedHeapSort(arr2, len);
    validateArray("Cache-blocked Heap Sort", arr2, len);
    printArray(arr2, len);

    free(arr1);
    free(arr2);
    return 0;
}
