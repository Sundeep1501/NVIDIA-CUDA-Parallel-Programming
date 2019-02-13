#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <iomanip>
using namespace std;

#define iceil(num,den) (num+den-1)/den

//Kernel Function
__global__ void imgMulKernel(float* d_img_in, float* d_img_out, int w, int h, float *d_img_fin, int fw, int fh){
	
	//Access the pixel on the image
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	
	//Ignore border elements
	if(r == 0 || r == h-1 || c == 0 || c == w-1)
		return;

	//Check if the index is in the image 
	if(c < w && r < h){
		int tl = d_img_in[(r-1)*w + (c-1)] * d_img_fin[0*fw + 0];
		int tc = d_img_in[(r-1)*w + (c)] * d_img_fin[0*fw + 1];
		int tr = d_img_in[(r-1)*w + (c+1)] * d_img_fin[0*fw + 2];

		int ml = d_img_in[r*w + (c-1)] * d_img_fin[1*fw + 0];
		int mc = d_img_in[r*w + c] * d_img_fin[1*fw + 1];
		int mr = d_img_in[r*w + (c+1)] * d_img_fin[1*fw + 2];
		
		int bl = d_img_in[(r+1)*w + (c-1)] * d_img_fin[2*fw + 0];
		int bc = d_img_in[(r+1)*w + (c)] * d_img_fin[2*fw + 1];
		int br = d_img_in[(r+1)*w + (c+1)] * d_img_fin[2*fw + 2];

		int sum = tl + tc + tr + ml + mc + mr + bl + bc + br;
		
		d_img_out[r*w + c] = sum/9;
		//d_img_out[row*w + col] = d_img_in[row*w + col] * 2;
	}
}

void imgMul(float*img_in, float* img_out, int w, int h, float *img_fin, int fw, int fh) {
	
	//This number of bytes are going be allocated and transferred
	int size = w * h * sizeof(float);
	int fsize = fw * fh * sizeof(float);

	float *d_img_in, *d_img_out, *d_img_fin;

	//Allocate memory from GPU for my input and output array
	cudaMalloc((void**)&d_img_in, size);
	cudaMalloc((void**)&d_img_out, size);
	cudaMalloc((void**)&d_img_fin, fsize);

	//Transfer data to the GPU
	cudaMemcpy(d_img_in, img_in, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_img_fin, img_fin, fsize, cudaMemcpyHostToDevice);

	dim3 myBlockDim(16, 16, 1);
	dim3 myGridDim(iceil(w, 16), iceil(h, 16), 1);
	//Kernel Launch
	imgMulKernel <<<myGridDim, myBlockDim >>> (d_img_in, d_img_out, w, h, d_img_fin, fw, fh);

	//Transfer results back to CPU
	cudaMemcpy(img_out, d_img_out, size, cudaMemcpyDeviceToHost);

	cudaFree(d_img_in);
	cudaFree(d_img_out);
}

std::string to_format(const int number) {
    std::stringstream ss;
    ss << std::setw(2) << std::setfill(' ') << number;
    return ss.str();
}

//Prints the image on screen
void printImage(float* img, int w, int h) {
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			cout << to_format(img[i*w + j]) << " ";
		}
		cout << endl;
	}
	cout << endl;
}

int main(int argc, char **argv){

	
	int w = atoi(argv[1]);
	int h = atoi(argv[2]);
	int fw = 3;
	int fh = 3;

	float *img_in = new float[w*h];
	float *img_out = new float[w*h];
	float *img_fin = new float[fw*fh];

	time_t t;
	srand((unsigned) time(&t));

	//Load value to the image
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			img_in[i*w + j] =rand() % 10 ;

	// Load value to filter
	for(int i = 0; i < fh; i++)
		for(int j = 0; j < fw; j++)
			img_fin[i*fw + j] = rand() % 10;
	
	cout << "Filter Array" << endl;
	printImage(img_fin, fw, fh);

	cout << "Input Array" << endl;
	printImage(img_in, w, h);

	imgMul(img_in, img_out, w, h, img_fin, fw, fh);
	
	cout << "Output Array" << endl;
	printImage(img_out, w, h);

	return 0;
}
