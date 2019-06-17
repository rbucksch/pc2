/**
 * Projekt Parallel Computing 2
 * Waermeleitung auf NxN-Gitter unter Verwendung von Cuda
 *
 * Autor: Robert Bucksch
 * Datum: 18.06.19
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// TDIM x TDIM Threads pro Block
#define TDIM 16 

// vertauscht zwei Pointer
__host__ __device__ void swap(double *&a, double *&b) {
  double* tmp = a;
  a = b;
  b = tmp;
}

// speichert Grenzen des Shared-Memory-Bereichs, verschiebt diese
// nach innen, falls Block einen Bereich am Rand des Gitters bearbeitet
struct bounds {
  __device__ bounds(int smdim, int gc) {
    top    = (blockIdx.y == 0 ? gc : 0);
    left   = (blockIdx.x == 0 ? gc : 0);
    right  = smdim - (blockIdx.x == gridDim.x - 1 ? gc : 0);
    bottom = smdim - (blockIdx.y == gridDim.y - 1 ? gc : 0);
  }
  int top, left, right, bottom;
};

// befuellt die beiden Zonen sm und tmp des Shared Memory mit Daten aus globalem Array t1
__device__ void fillSM(double *sm, double *gm, int smdim, int gmdim, int gc,
                       bounds b) {
  // Offset von Shared Memory bzgl globalem Array gm berechnen
  int gOS = blockDim.x * (blockDim.y * blockIdx.y * gridDim.x+ blockIdx.x)
            - gc * gmdim - gc;

  for (int i = threadIdx.y + b.top; i < b.bottom; i += blockDim.y) {
    for (int j = threadIdx.x + b.left; j < b.right; j += blockDim.x) {
      double s = gm[gOS + i * gmdim + j];
      // schreibe in sm
      sm[i * smdim + j] = s;
      // schreibe in tmp
      sm[i * smdim + j + smdim * smdim] = s;
    }
  }
}

// schreibt die finalen Daten im Shared Memory an die entsprechenden Stellen in t2
__device__ void write_back(double *sm, double *gm, int smdim, int gmdim, int gc) {
  // Offset in global memory berechnen
  int gOS = blockDim.x * (blockDim.y * blockIdx.y * gridDim.x+ blockIdx.x);
  // Offset in shared memory berechnen
  int sOS = gc * smdim + gc;

  gm[gOS + threadIdx.y * gmdim + threadIdx.x] = sm[sOS + threadIdx.y * smdim + threadIdx.x];
}

__device__ void update_sm(double *sm, double *tmp, int smdim, bounds b) {
  for (int i = threadIdx.y + b.top + 1; i < b.bottom - 1; i += blockDim.y) {
    for (int j = threadIdx.x + b.left + 1; j < b.right - 1; j += blockDim.x) {
      tmp[i * smdim + j] = 0.2 * (  sm[(i - 1) * smdim + j]
                                  + sm[i * smdim + (j - 1)]
                                  + sm[i * smdim + j]
                                  + sm[i * smdim + (j + 1)]
                                  + sm[(i + 1) * smdim + j]);
    }
  }
}

__global__ void update(double *t1, double *t2, int size, int gc, int smdim) {
  extern __shared__ double shared[];

  // teile Shared Memory in zwei Bereiche auf
  double* sm = &shared[0];
  double* tmp = &shared[smdim*smdim];

  bounds b(smdim, gc);

  // shared memory befuellen
  fillSM(shared, t1, smdim, size, gc, b);
  __syncthreads();

  // sm so oft updaten, wie es die Breite der Geisterzone erlaubt
  for (int k = 0; k < gc; ++k) {
    update_sm(sm, tmp, smdim, b);
    __syncthreads();
    swap(sm, tmp);
  }

  // Daten zurueckschreiben
  write_back(sm, t1, smdim, size, gc);
}

// Initialisiere Wärmefeld mit Startwerten:
// innen: 0.0
// Rand:
// links/oben warm=25.0
// rechts/unten kalt=-25.0
void init(double *t, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      t[j + i * size] = 0.0;
      if (j == 0)
        t[i * size] = 25.0;
      if (j == size - 1)
        t[j + i * size] = -25.0;
      if (i == 0)
        t[j + i * size] = 25.0;
      if (i == size - 1)
        t[j + i * size] = -25.0;
    }
  }
}

// Ausgabe des Feldes t als PPM (Portable Pix Map) in filename
// mit schönen Farben
void printResult(double *t, int size, char *filename) {
  FILE *f = fopen(filename, "w");
  fprintf(f, "P3\n%i %i\n255\n", size, size);
  double tmax = 25.0;
  double tmin = -tmax;
  double r, g, b;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      double val = t[j + i * size];
      r = 0;
      g = 0;
      b = 0;
      if (val <= tmin) {
        b = 1.0 * 255.0;
      } else if (val >= -25.0 && val < -5) {
        b = 255 * 1.0;
        g = 255 * ((val + 25) / 20);
      } else if (val >= -5 && val <= 0.0) {
        g = 255 * 1.0;
        b = 255 * (1.0 - (val + 5) / 5);
      } else if (val > 0.0 && val <= 5) {
        g = 255 * 1.0;
        r = 255 * ((val) / 5);
      } else if (val > 5 && val < 25.0) {
        r = 255 * 1.0;
        g = 255 * ((25 - val) / 20);
      } else {
        r = 255 * 1.0;
      }
      fprintf(f, "%i\n%i\n%i\n", (int)r, (int)g, (int)b);
    }
    //      fprintf(f,"\n");
  }
  fclose(f);
}

int main(int argc, char **argv) {
  // Größe des Feldes
  int size = 128;
  // Breite der Geisterzone
  int ghostcells = 2;
  // Anzahl Iterationen
  int iter = 100;
  // Ausgabedatei
  char *filename = (char*)"out.ppm";

  // Übergabeparameter für [size ghostcells iter filename] einlesen
  if (argc > 1) size = atoi(argv[1]);
  if (argc > 2) ghostcells = atoi(argv[2]);
  if (argc > 3) iter = atoi(argv[3]);
  if (argc > 4) filename = argv[4];

  if (ghostcells < 1) {
    printf("Error: minimal width of ghostzone is 1\n");
    printf("Usage: %s [size] [ghostcells] [iter] [filename]\n", argv[0]);
    return -1;
  } else if (size % TDIM != 0) {
    printf("Error: size must be a multiple of %d\n", TDIM);
    printf("Usage: %s [size] [ghostcells] [iter] [filename]\n", argv[0]);
    return -1;
  }

  // Cache Config
  cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);

  // 2 Speicherbereiche für das Wärmefeld auf Host und Device
  double *t1_host, *t2_host;
  double *t1_dev, *t2_dev;

  // Größe des Speicherbereiches
  int mem = size * size * sizeof(double);
  // Allokiere Speicher auf Host
  t1_host = (double *)malloc(mem);
  t2_host = (double *)malloc(mem);
  // Initialisiere Speicher
  init(t1_host, size);

  // CUDA Speicher anlegen
  cudaMalloc((void **)&t1_dev, mem);
  cudaMalloc((void **)&t2_dev, mem);

  // Host->Device Memcpy von t1
  cudaMemcpy(t1_dev, t1_host, mem, cudaMemcpyHostToDevice);

  // 2D Threads/Grid anlegen
  int tdim = min(TDIM, size);
  dim3 threads(tdim, tdim);
  dim3 grid(size / threads.x, size / threads.y);

  // Wiederhole update Kernel für iter Iterationen
  for (int iters_left = iter; iters_left > 0; iters_left -= ghostcells) {
    int gc = min(iters_left, ghostcells);
    int smdim = tdim + 2 * gc;
    int smem = smdim * smdim * sizeof(double);
    update<<<grid, threads, 2*smem>>>(t1_dev, t2_dev, size, gc, smdim);
  }

  // Kopiere Endzustand zurück
  cudaMemcpy(t2_host, t1_dev, mem, cudaMemcpyDeviceToHost);

  // Ausgabe des Endzustandes mit printResult
  printResult(t2_host, size, filename);

  // Speicher Freigeben (Device+Host)
  free(t1_host);
  free(t2_host);
  cudaFree(t1_dev);
  cudaFree(t2_dev);

  return 0;
}
