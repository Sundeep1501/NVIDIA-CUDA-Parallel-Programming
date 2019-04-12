≈#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <iomanip>
using namespace std;

#define iceil(num,den) (num+den-1)/den

//Kernel Function
__global__ void matrixMulKernel(float* d_matrix1, int r1, int c1, float* d_matrix2, int r2, int c2, float *d_matrixR){
	
	//Access the pixel on the image
	int r = blockDim.x * blockIdx.x + threadIdx.x;
	int c = blockDim.y * blockIdx.y + threadIdx.y;

	//Check if the index is in the image 
	if(c < c2 && r < r1){
		
		int value = 0;
		for(int i = 0; i < r2; i++){
			value+= d_matrix1[r*c1+i] * d_matrix2[i*c2+c];
		}
		d_matrixR[r*c2 + c] = value;
	}
}

void matrixMul(float* matrix1, int r1, int c1, float* matrix2, int r2, int c2, float* matrixR) {
	
	//This number of bytes are going be allocated and transferred
	int size1 = r1 * c1 * sizeof(float);
	int size2 = r2 * c2 * sizeof(float);
	int sizeR = r1 * c2 * sizeof(float);

	float *d_matrix1, *d_matrix2, *d_matrixR;

	//Allocate memory from GPU for my input and output array
	cudaMalloc((void**)&d_matrix1, size1);
	cudaMalloc((void**)&d_matrix2, size2);
	cudaMalloc((void**)&d_matrixR, sizeR);

	//Transfer data to the GPU
	cudaMemcpy(d_matrix1, matrix1, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrix2, matrix2, size2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixR, matrixR, sizeR, cudaMemcpyHostToDevice);

	dim3 myBlockDim(16, 16, 1);
	dim3 myGridDim(iceil(r1, 16), iceil(c2, 16), 1);
	//Kernel Launch
	matrixMulKernel <<<myGridDim, myBlockDim >>> (d_matrix1, r1, c1, d_matrix2, r2, c2, d_matrixR);

	//Transfer results back to CPU
	cudaMemcpy(matrixR, d_matrixR, sizeR, cudaMemcpyDeviceToHost);

	// free cuda memory
	cudaFree(d_matrix1);
	cudaFree(d_matrix2);
	cudaFree(d_matrixR);
}

std::string to_format(const int number) {
    std::stringstream ss;
    ss << std::setw(2) << std::setfill(' ') << number;
    return ss.str();
}

//Prints the image on screen
void printImage(float* img, int r, int c) {
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			cout << to_format(img[i*c + j]) << " ";
		}
		cout << endl;
	}
	cout << endl;
}

int main(int argc, char **argv){

	int r1 = atoi(argv[1]);
	int c1 = atoi(argv[2]);
	int r2 = atoi(argv[3]);
	int c2 = atoi(argv[4]);

	// basic matrix-matrix multiplication rule
	if (c1 != r2){
		cout << "#columns of first matrix and #rows of second matrix should be equal.";
		return 0;
	}

	float *matrix1 = new float[r1*c1];
	float *matrix2 = new float[r2*c2];
	float *matrixR = new float[r1*c2];

	time_t t;
	srand((unsigned) time(&t));

	//Load value to the image
	for (int i = 0; i < r1; i++)
		for (int j = 0; j < c1; j++)
			matrix1[i*c1 + j] = rand() % 10 ;

	// Load value to filter
	for(int i = 0; i < r2; i++)
		for(int j = 0; j < c2; j++)
			matrix2[i*c2 + j] = rand() % 10;
	
	cout << "Matrix1 Array" << endl;
	printImage(matrix1, r1, c1);

	cout << "Matrix2 Array" << endl;
	printImage(matrix2, r2, c2);

	matrixMul(matrix1, r1, c1, matrix2, r2, c2, matrixR);
	
	cout << "Result Array" << endl;
	printImage(matrixR, r1, c2);

	return 0;
}
