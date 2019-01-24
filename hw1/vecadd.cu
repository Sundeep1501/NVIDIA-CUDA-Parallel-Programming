// very simple vector add example discussed in class
// --everything is in one *.cpp program
// --no error checking; not a good idea

using namespace std;
#include <iostream>

#define TILE_WIDTH 256

// iceil macro
// returns an integer ceil value where integer numerator is first parameter
// and integer denominator is the second parameter. iceil is the rounded
// up value of numerator/denominator when there is a remainder
// equivalent to ((num%den!=0) ? num/den+1 : num/den)
#define iceil(num,den) (num+den-1)/den 

// Basic CPU based vecAdd C = A+B implementation
// Replaced by GPU function with identical name
// (commented out here!)
/*
void vecAdd (float *A, float *B, float *C, int n) {
   int i;
   for (i=0;i<n;i++) C[i] = A[i] + B[i];
}
*/


// GPU Vector Add kernel
__global__ void vectAddKernel(float *A, float *B, float *C, int N) {
   int i = blockDim.x*blockIdx.x+threadIdx.x;
   
   if (i<N) C[i] = A[i]+B[i];
}

// GPU Vector Add Function
void vecAdd(float *A, float *B, float *C, int N) {

   int size = N*sizeof(float);

   // allocate device (GPU) memory
   float *d_A,*d_B,*d_C;

   // Allocate device memory and Transfer host arrays A and B 
   cudaMalloc((void **) &d_A,  size);
   cudaMemcpy(d_A, A,  size, cudaMemcpyHostToDevice);

   cudaMalloc((void **) &d_B,  size);
   cudaMemcpy(d_B, B,  size, cudaMemcpyHostToDevice);

   // Allocate device memory of P array for results
   cudaMalloc((void **) &d_C,  size);

   // Setup the kernel execution configuration parameters
   // & Launch Kernel!
   vectAddKernel <<<iceil(N,TILE_WIDTH),TILE_WIDTH>>> (d_A,d_B,d_C,N);
   cudaError_t error_id=cudaGetLastError();
   if (error_id != cudaSuccess) {
      cout << "Attempted Launch of MatriMulKernel returned " << 
          (int)error_id  << endl;
      cout <<  cudaGetErrorString(error_id) << endl ;
      exit(EXIT_FAILURE);
   }

   // Transfer Vector C from device to host
   cudaMemcpy(C,d_C, size,cudaMemcpyDeviceToHost);

   // Free device matrices
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);
}


int main (int argc, char **argv) {
  
   int N;

   if (argc != 2) {
      cout << "usage: vectoradd <vector size>" << endl;
      exit(1);
   }

   N = atoi(argv[1]);
 
   // allocate and initialize host (CPU) memory
   float *A = new float[N];   
   float *B = new float[N]; 
   float *C = new float[N];

   // intialize the two vectors
   for (int i=0;i<N;i++) A[i]=B[i]=(float) i;

   // Execute Vector Add Functions
   vecAdd(A, B, C, N);

   // Output Results
   for (int i=0;i<N;i++) cout << "C[" << i << "]=" << C[i] << endl;

   free(A); free(B); free(C);
}


