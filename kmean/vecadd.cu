// very simple vector add example discussed in class
// --everything is in one *.cpp program
// --no error checking; not a good idea

using namespace std;
#include <iostream>
#include <math.h>

#define TILE_WIDTH 256

// iceil macro
// returns an integer ceil value where integer numerator is first parameter
// and integer denominator is the second parameter. iceil is the rounded
// up value of numerator/denominator when there is a remainder
// equivalent to ((num%den!=0) ? num/den+1 : num/den)
#define iceil(num,den) (num+den-1)/den 

// GPU kernel
__global__ void findNearestCentroidKernel(float *X, float *Y, float *CX, float *CY, int *CI, int N, int K) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < N) {
		float minDis = sqrt(((CX[0] - X[i])*(CX[0] - X[i])) + ((CY[0] - Y[i])*(CY[0] - Y[i])));
		int nearestIndex = 0;
		for(int k = 1; k < K; k++){
			// find nearest centroid index for this point
			float dis = sqrt(((CX[k] - X[i])*(CX[k] - X[i])) + ((CY[k] - Y[i])*(CY[k] - Y[i])));
			if(dis < minDis){
				minDis = dis;
				nearestIndex = k;
			}
		}

		CI[i] = nearestIndex;
	}
}

void findNearestCentroid(float *X, float *Y, float *CX, float *CY, int *CI, int N, int K) {
	int size = N * sizeof(float);
	int kSize = K * sizeof(float);
	int iSize = N * sizeof(int);

	float *d_X, *d_Y, *d_CX, *d_CY;
	int  *d_CI;
	
	// allocate device memory and transfer points and centroids
	cudaMalloc((void **) &d_X, size);
	cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &d_Y, size);
	cudaMemcpy(d_Y, Y, size, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &d_CX, kSize);
	cudaMemcpy(d_CX, CX, kSize, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_CY, kSize);
	cudaMemcpy(d_CY, CY, kSize, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_CI, iSize);
	cudaMemcpy(d_CI, CI, iSize, cudaMemcpyHostToDevice);

	findNearestCentroidKernel <<<iceil(N,TILE_WIDTH),TILE_WIDTH>>>(d_X, d_Y, d_CX, d_CY, d_CI, N, K);

        cudaError_t error_id=cudaGetLastError();
        if (error_id != cudaSuccess) {
                cout << "Attempted Launch of MatriMulKernel returned " <<
                (int)error_id  << endl;
                cout <<  cudaGetErrorString(error_id) << endl ;
                exit(EXIT_FAILURE);
        }


	cudaMemcpy(CI, d_CI, iSize, cudaMemcpyDeviceToHost);
	
	// free device memory
	cudaFree(d_X); cudaFree(d_Y); cudaFree(d_CX); cudaFree(d_CY); cudaFree(d_CI);
}

int main (int argc, char **argv) {
   int N = 12;  
   int K = 2; // there are two clusters

   // allocate and initialize host (CPU) memory
   float xp[] = {2, 3, 4, 2, 4, 2, 3, 4, 7, 8, 7, 8};
   float yp[] = {2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6};    
   
   float *X = new float[N];   
   float *Y = new float[N]; 
   float *CX = new float[K];
   float *CY = new float[K];

   for (int i=0;i<N;i++) {
	X[i] = xp[i];
	Y[i] = yp[i];
   }
   // selecting first and last point as random centroids
   CX[0] = X[0];
   CY[0] = Y[0];
   
   CX[1] = X[7];
   CY[1] = Y[7];
   
   int *CI = new int[N];
   
	int count[K];
	float xDisSum[K];
	float yDisSum[K];
	for(int i=0; i<K; i++) {
		xDisSum[i] = 0;
		yDisSum[i] = 0;
		count[i] = 0;
	}
   	
	while(true){

		float *NCX = new float[K];
		float *NCY = new float[K];

   		findNearestCentroid(X, Y, CX, CY, CI, N, K);
	
		for(int i=0; i<N; i++){
			xDisSum[CI[i]] += X[i];
			yDisSum[CI[i]] += Y[i];
			count[CI[i]]++;
		}
		
		int isMoved = 0;
		// calculate mean distances and update centroids
		for(int	i=0; i<K; i++) {
         		float nx = xDisSum[i]/count[i];
                	float ny = yDisSum[i]/count[i];
			
			if (nx!=CX[i] || ny!=CY[i]){
				isMoved = 1;
				CX[i] = nx;
				CY[i] = ny;
			}
			
			cout << "CX[" << i << "]=" << CX[i] << endl;
			cout <<	"CY[" << i << "]=" << CY[i] << endl;
        	}
		
		
		if(isMoved == 0){
			break;	
		}
		
   	}
   // Output Results
    // for (int i=0;i<N;i++) cout << "CI[" << i << "]=" << CI[i] << endl;

   free(X); free(Y); free(CX); free(CY); free(CI);
}
