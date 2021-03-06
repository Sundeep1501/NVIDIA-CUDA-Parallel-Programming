// very simple vector add example discussed in class
// --everything is in one *.cpp program
// --no error checking; not a good idea

using namespace std;
#include <iostream>
#include <math.h>
#include <stdio.h>

#define TILE_WIDTH 512

// iceil macro
// returns an integer ceil value where integer numerator is first parameter
// and integer denominator is the second parameter. iceil is the rounded
// up value of numerator/denominator when there is a remainder
// equivalent to ((num%den!=0) ? num/den+1 : num/den)
#define iceil(num,den) (num+den-1)/den 

// GPU kernel
__global__ void findNearestCentroidKernel(float *X, float *Y, float *CX, float *CY, float *TCX, float *TCY, int *COUNT, int N, int K, int *BREAK, float *R) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

   while(BREAK[0] == 0){
	if(i < K){
		R[i] = 0;
		TCX[i] = 0;
		TCY[i] = 0;
		COUNT[i] = 0;
	}
	__syncthreads();

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
		
		// radius	
		float maxDis = sqrt(((CX[nearestIndex] - X[i])*(CX[nearestIndex] - X[i])) + ((CY[nearestIndex] - Y[i]) * (CY[nearestIndex] - Y[i])));
		if(R[nearestIndex] < maxDis){
			R[nearestIndex] = maxDis;
		}

		// sum to calculate mean
		atomicAdd(&TCX[nearestIndex], X[i]);
		atomicAdd(&TCY[nearestIndex], Y[i]);
		atomicAdd(&COUNT[nearestIndex], 1);
	}
	__syncthreads();

	if(i < K){
		TCX[i] /= COUNT[i];
		TCY[i] /= COUNT[i];
	}
	__syncthreads();
	
	
	if(i==0) {
		int isMoved = 0;
		for(int k = 0; k < K; k++) {
			if(CX[k] != TCX[k] || CY[k] != TCY[k]){
		    		isMoved = 1;
			}

			CX[k] = TCX[k];
			CY[k] = TCY[k];
		}
	
		if(isMoved == 0){
			BREAK[0] = 1;
		}
	}
	__syncthreads();
   }
}

void findNearestCentroid(float *X, float *Y, float *CX, float *CY, float *TCX, float *TCY, int *COUNT, int N, int K, int *BREAK, float *R) {
	int size = N * sizeof(float);
	int kSize = K * sizeof(float);

	float *d_X, *d_Y, *d_CX, *d_CY, *d_TCX, *d_TCY, *d_R;
	int *d_COUNT, *d_BREAK;
	
	// allocate device memory and transfer points and centroids
	cudaMalloc((void **) &d_X, size);
	cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &d_Y, size);
	cudaMemcpy(d_Y, Y, size, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &d_CX, kSize);
	cudaMemcpy(d_CX, CX, kSize, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_CY, kSize);
	cudaMemcpy(d_CY, CY, kSize, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_TCX, kSize);
	cudaMemcpy(d_TCX, TCX, kSize, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &d_TCY, kSize);
        cudaMemcpy(d_TCY, TCY, kSize, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_COUNT, K * sizeof(int));
        cudaMemcpy(d_COUNT, COUNT, K * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_BREAK, sizeof(int));
	cudaMemcpy(d_BREAK, BREAK, sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_R, kSize);
        cudaMemcpy(d_R, R, kSize, cudaMemcpyHostToDevice);

	findNearestCentroidKernel <<<iceil(N,TILE_WIDTH),TILE_WIDTH>>>(d_X, d_Y, d_CX, d_CY, d_TCX, d_TCY, d_COUNT, N, K, d_BREAK, d_R);

        cudaError_t error_id=cudaGetLastError();
        if (error_id != cudaSuccess) {
                cout << "Attempted Launch of MatriMulKernel returned " <<
                (int)error_id  << endl;
                cout <<  cudaGetErrorString(error_id) << endl ;
                exit(EXIT_FAILURE);
        }


	cudaMemcpy(CX, d_CX, kSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(CY, d_CY, kSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(TCX, d_TCX, kSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(TCY, d_TCY, kSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(BREAK, d_BREAK, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(R, d_R, kSize, cudaMemcpyDeviceToHost);
	
	// free device memory
	cudaFree(d_X); cudaFree(d_Y); cudaFree(d_CX); cudaFree(d_CY);
	cudaFree(d_TCX); cudaFree(d_TCY); cudaFree(d_COUNT); cudaFree(d_BREAK);
	cudaFree(d_R);
}

int main (int argc, char **argv) {
   	// code test input
   	// float xp[] = {2, 3, 4, 2, 4, 2, 3, 4, 7, 8, 7, 8};
   	// float yp[] = {2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6};
	// int K = 2;    
   
	// dataset
	float xp[] = {0.50045,0.62042,0.63662,0.78343,0.39833,0.76105,0.86649,0.5338,0.67921,0.58146,0.51889,0.82518,0.38533,0.53911,0.90309,0.94548,0.79293,0.7579,0.28621,0.44331,0.69774,0.4574,0.70613,0.72721,0.83308,0.47526,0.53495,0.99023,0.45525,0.85482,0.77327,0.86059,0.74007,0.74936,0.59231,0.55414,0.84518,0.74866,0.49398,0.67464,0.80713,0.58186,0.43138,0.79422,0.83607,0.70866,0.91123,0.91224,0.89366,0.69314,0.77839,0.79542,0.62412,0.42176,0.77219,0.45899,0.597,0.72932,0.89745,0.86918,0.86877,0.60436,0.44866,0.67697,0.48163,0.7069,0.64928,0.69971,0.37205,0.63079,0.61336,0.35087,0.50331,0.84022,0.30077,0.78189,0.87846,0.52592,0.97682,0.36551,0.58415,0.68087,0.49863,0.47702,0.80379,0.84552,0.82612,0.58831,0.71168,0.47222,0.98558,0.82497,0.35571,0.77466,0.53953,0.45861,0.68372,0.41362,0.27898,0.71739,-0.61555,-0.059872,0.15817,-0.80462,0.013774,-0.026154,0.57119,0.1383,0.14187,-0.88263,0.59423,-0.44384,0.2998,-0.24294,0.13261,0.46752,-0.14789,0.43184,0.17082,0.60789,-0.36113,-0.2566,-0.99087,-0.53892,0.33925,-0.49112,0.16715,-0.019133,0.0705,0.46192,-0.77776,-0.62921,0.32942,0.016021,-0.21196,-0.61637,0.53244,0.38571,-0.47271,0.080991,0.039742,-0.65584,0.57717,-0.57962,-0.015825,-0.91257,-0.64211,-0.72704,0.24372,-0.21568,-0.4985,-0.2709,0.16699,0.01051,-0.30193,0.18312,-0.87527,0.42738,-0.21141,0.41621,0.11842,-0.14174,-0.18126,-0.098237,-0.3365,0.14356,0.35141,0.13923,-0.16566,-0.70101,-0.0694,-0.68869,-0.38079,-0.43985,0.14795,-0.3374,-0.89033,0.15952,0.0018652,0.45042,0.080862,-0.054846,-0.50866,-0.66301,-0.099192,0.053784,-0.15054,-0.60872,-1.0649,-0.76489,-0.23357,-0.56653,0.22882,0.036619,-0.47542,0.53557,0.057822,-0.21597,0.084768,0.29547,-1.6625,-2.3066,-2.5451,-2.5055,-1.8161,-2.2007,-1.5987,-2.2245,-1.8461,-2.331,-1.6917,-1.5426,-2.0721,-1.5946,-2.5846,-1.9467,-2.2048,-1.7053,-2.2967,-2.1201,-2.0924,-1.8321,-2.0276,-1.7004,-2.1879,-2.1517,-1.7454,-2.3393,-2.3783,-2.028,-1.77,-1.8245,-1.9501,-1.8441,-1.4555,-2.7113,-2.1346,-1.8869,-1.596,-2.0089,-2.4819,-2.0069,-2.7124,-2.0677,-2.3332,-1.8419,-2.0059,-1.9892,-2.4431,-2.5464,-2.3451,-2.2541,-2.5845,-1.5723,-1.7045,-1.5421,-2.5846,-1.8773,-2.2585,-2.407,-1.9173,-2.5959,-1.8451,-2.0366,-2.3052,-1.7897,-2.4067,-1.4965,-2.1428,-2.0735,-2.3237,-2.4941,-1.7384,-1.7941,-1.7079,-2.4116,-2.4062,-2.3151,-2.6612,-1.7589,-2.5339,-1.531,-2.1624,-1.8302,-1.5692,-1.7906,-2.2307,-2.1941,-2.5677,-2.374,-1.9896,-2.006,-2.3813,-2.1396,-1.989,-2.4698,-1.519,-1.4752,-1.8924,-1.9298};
	float yp[] = {0.99137,1.1864,1.1091,1.1882,0.86828,1.2912,1.1251,1.2909,1.2921,0.91792,0.98102,0.87589,1.0807,0.879,1.1005,1.0128,1.1342,1.1215,1.0179,1.0879,0.65743,0.8492,1.2998,0.891,0.85405,0.75205,1.0531,0.83313,0.80712,0.65705,1.2371,0.82968,1.1268,1.082,1.252,1.0891,1.0913,1.248,1.2467,0.71219,0.8547,1.0433,1.141,1.2024,0.68393,1.2528,0.96512,0.89731,0.92602,0.77342,1.2817,1.1748,1.2687,0.76846,0.91627,1.0187,1.1778,0.97085,1.1594,0.71518,1.263,0.85016,0.71094,1.0995,1.1069,0.72923,0.90394,0.67114,0.94768,1.1872,0.94457,0.83987,1.1424,1.2836,0.91407,1.1718,0.80268,0.60764,0.81599,0.91851,1.324,1.2934,1.3125,1.1482,0.95867,1.2578,0.63838,1.3116,0.92079,0.99451,0.93409,1.0769,0.74702,0.99721,1.3215,0.98109,1.0413,0.7791,0.9537,0.62689,-0.44045,0.44408,-0.24966,0.5796,-0.50121,0.63359,0.13374,-0.24062,-0.53195,0.64043,0.57053,0.0097233,0.16545,0.2039,0.74279,-0.39876,0.31784,0.20217,-0.25401,0.058463,-0.57635,0.0088247,-0.054508,-0.041507,0.63631,-0.46832,-0.56195,-0.63778,0.40821,-0.20645,0.85026,0.99861,0.041061,0.34774,-0.21727,0.58605,0.66522,0.11627,-0.31435,-0.36548,0.13246,-0.27425,0.33963,0.20461,0.66094,-0.40905,0.59764,-0.36371,-0.47063,-0.52933,0.47001,0.07556,0.32864,-0.44967,-0.3139,0.28728,0.18796,-0.12,1.026,-0.14989,0.67563,0.8246,0.12724,-0.40327,-0.15703,-0.32224,0.80277,0.77583,0.37448,0.096927,0.047256,-0.092455,0.50284,-0.31497,0.3849,0.68048,-0.24552,0.92299,1.0894,-0.35057,-0.096622,0.58476,1.0457,0.80786,0.4435,0.77992,-0.16127,-0.57881,-0.018881,-0.32336,0.34652,1.0319,-0.39581,0.29568,0.61551,-0.091232,0.49379,0.049174,0.33141,0.019771,0.39901,-0.28535,-0.14886,-0.36566,-0.60674,-0.027574,0.31511,-0.40363,-0.40677,0.29229,-0.0034327,0.046595,0.67712,-0.16778,0.39392,0.11283,0.47853,0.55313,0.26466,-0.15206,-0.53149,0.22226,0.37138,-0.053052,-0.52837,0.055299,0.3594,0.078523,-0.25653,-0.13005,0.61349,-0.42266,0.60408,-0.12923,-0.17669,-0.010072,0.17846,0.39363,0.23257,0.033562,-0.50101,-0.46628,0.05617,0.32735,-0.29258,-0.30209,0.071615,-0.18159,-0.10489,-0.032858,-0.12095,0.47737,0.34116,-0.31788,-0.36801,-0.26249,-0.23782,0.36064,-0.46076,0.46653,0.23937,0.33773,-0.56519,0.65367,-0.46669,0.48537,0.1671,0.3639,0.33264,-0.22258,0.25899,0.18506,0.013162,0.44468,0.18279,-0.56412,0.15889,-0.35779,-0.074785,0.29244,0.23565,-0.024191,0.11848,-0.32517,0.14238,0.5062,-0.66232,0.5214,0.27723,-0.077432,-0.30826,0.33229,-0.35625,-0.53503,0.21018,0.025991,-0.45004,-0.005085,-0.59767,0.1629};
	int K = 3;

	int N = sizeof(xp)/sizeof(int); //length of an integer array
	
	float *X = new float[N];   
  	float *Y = new float[N]; 
   	float *CX = new float[K];
   	float *CY = new float[K];
   	float *TCX = new float[K];
   	float *TCY = new float[K];
	float *R = new float[K];
	int *COUNT = new int[K];
	int *BREAK = new int[1];

   	for (int i=0;i<N;i++) {
		X[i] = xp[i];
		Y[i] = yp[i];
   	}
   
	for(int k = 0; k < K; k++) {
		TCX[k] = 0;
		TCY[k] = 0;
		COUNT[k] = 0;
		R[k] = 0;

		// random centers
		int randomIndex = rand() % N;
		CX[k] = X[randomIndex];
		CY[k] = Y[randomIndex];
	}
	BREAK[0] = 0;

   	findNearestCentroid(X, Y, CX, CY, TCX, TCY, COUNT, N, K, BREAK, R);
			
   	// Output Results
   	for (int i=0;i<K;i++) {
		cout << "CX[" << i << "]=" << CX[i] <<", " << "CY[" << i << "]=" << CY[i] <<", Radius[" << i << "]="<<R[i]<< endl;
	}
	cout << "BREAK=" << BREAK[0] << endl;

   	free(X); free(Y); free(CX); free(CY); free(TCX); free(TCY); free(BREAK); free(COUNT); free(R);
}
