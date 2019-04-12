#include <iostream>

using namespace std;

#define TILE_WIDTH 64
#define iceil(num,den) (num+den-1)/den

//Prints the image on screen
void printMatrix(float* img, int w, int h) {
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			cout << img[i*w + j] << " ";
		}
		cout << endl;
	}
	cout <<"*****" << endl;
}  

__global__ void matrixMulKernel_NoSM(float *d_M, float *d_N, float *d_P, int width){ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(col < 1 && row < width){
	float pValue = 0;
        for(int k = 0; k < width; k++) 
        {
            pValue += d_M[row*width + k] * d_N[k + col];
        }
        d_P[row + col] = pValue;
    }
}

__global__ void matrixMulKernel_SM(float *d_M, float *d_N, float *d_P, int width){ 
    
	__shared__ float Ms[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Ns[TILE_WIDTH][1];
	
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	
	int row = by * TILE_WIDTH + ty; 
    	int col = bx + tx; 
    
	float pValue = 0;
	
	for(int m=0; m<width/TILE_WIDTH; m++){
		Ms[ty][tx] = d_M[(m*TILE_WIDTH + tx) + row*width];
		Ns[ty][tx] = d_N[(m*TILE_WIDTH + ty) + col];
		__syncthreads();
		
        for(int k = 0; k < TILE_WIDTH; k++)
            pValue += Ms[ty][k] * Ns[k][tx];

		__syncthreads();
    	}
	d_P[row + col] = pValue;
}

void matrixMul(float* M, float* N, float* P, int width) {
	
	//This number of bytes are going be allocated and transferred
	int size = width * width * sizeof(float);
	int rsize = width * sizeof(float);

	float *d_M, *d_N, *d_P; //Device Pointers

	//Allocate memory from GPU for my input and output array
	cudaMalloc((void**)&d_M, size);
	cudaMalloc((void**)&d_N, rsize);
	cudaMalloc((void**)&d_P, rsize);

	//Transfer data to the GPU
	cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, N, rsize, cudaMemcpyHostToDevice);
	
	dim3 myBlockDim(1, TILE_WIDTH, 1);
	dim3 myGridDim(iceil(width, TILE_WIDTH), iceil(width, TILE_WIDTH), 1);
	
	//===== Not using Shared Memory ===============
	matrixMulKernel_NoSM <<<myGridDim, myBlockDim >>> (d_M, d_N, d_P, width);
	cudaMemcpy(P, d_P, rsize, cudaMemcpyDeviceToHost);
	
	printMatrix(P, 1, width);

	//===== Using Shared Memory ===================
	matrixMulKernel_SM <<<myGridDim, myBlockDim >>> (d_M, d_N, d_P, width);
	cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);
	
	printMatrix(P, 1, width);
	
	//----------------------------------------------------
	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);
}

int main(){

	srand(time(0));

	int width = 320;
	float *M = new float[width*width];
	float *N = new float[width];
	float *P = new float[width];

	//Load value to the image
	for (int i = 0; i < width; i++)
		for (int j = 0; j < width; j++)
			M[i*width + j] = rand()%10;

	for (int i = 0; i < width; i++)
		N[i] = rand()%10;

	printMatrix(M, width, width);
	printMatrix(N, 1, width);

	matrixMul(M, N, P, width);

	return 0;
}
