#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include<iostream>
#include <cstdio> // For perror
#include<complex>

//custom power function provides better accuracy for large exponents
int int_pow(int base, int exp) {
    int result = 1;
    while (exp > 0) {
        if (exp % 2 == 1) {
            result *= base;
        }
        base *= base;
        exp /= 2;
    }
    return result;
}

__device__ int device_int_pow(int base, int exp) {
    int result = 1;
    while (exp > 0) {
        if (exp % 2 == 1) {
            result *= base;
        }
        base *= base;
        exp /= 2;
    }
    return result;
}

__global__ void FORWARD_GPU(float *a, float *b, float *c, float *d, float *x,int N,int n_eqns,int step)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;	

	int iLeft,iRight,Delta;
	float k1,k2;

	if(threadId>=n_eqns) return;
	int iEven =  (device_int_pow(2,step+1) * (threadId + 1)) - 1; //even indices to be modified
	
	Delta = device_int_pow(2,step);
	iLeft = iEven - Delta;
	iRight = iEven + Delta;
    k1 = a[iEven] / b[iLeft];
    //accounting for the boundary
    if(iRight>=N){
        iRight = 0;
        k2 = 0;
    }
    else{
        k2 = c[iEven] / b[iRight];
    }

    b[iEven] = b[iEven] - k1 *c[iLeft]  - k2* a[iRight];
    d[iEven] = d[iEven] -  k1* d[iLeft] - k2* d[iRight];
    a[iEven] = -k1* a[iLeft];
    c[iEven] = -k2* c[iRight];

}

__global__ void BACKWARD_GPU(float *a, float *b, float *c, float *d, float *x,int N,int n_eqns,int step)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int iLeft,iRight,Delta;

    //we need no more threads per step than the number of equations
	if(threadId>=n_eqns) return;

	int iKnown =  (device_int_pow(2,step+1) * (threadId + 1)) - 1; 
	Delta = device_int_pow(2,step);

	iLeft = iKnown - Delta;
	iRight = iKnown + Delta;
    //exit loop upon hitting the boundary
    if(iRight>=N){
        return;}

	x[iLeft] = (d[iLeft] - a[iLeft]*x[iLeft-Delta] - c[iLeft]*x[iLeft+Delta])/b[iLeft];
	x[iRight] = (d[iRight] - a[iRight]*x[iRight-Delta] - c[iRight]*x[iRight+Delta])/b[iRight]; 
}

__global__ void SOLVE_EQ_GPU(float*a, float *b, float*c,float *d, float *x, int iMiddle, int iFinal) 
{
    x[iMiddle] = (d[iMiddle] - c[iMiddle]*d[iFinal]/b[iFinal])/(b[iMiddle]-c[iMiddle]*a[iFinal]/b[iFinal]);
	x[iFinal] = (d[iFinal]-a[iFinal]*d[iMiddle]/b[iMiddle])/(b[iFinal]-a[iFinal]*c[iMiddle]/b[iMiddle]);
}

//function to implement CR on CPU
double CR_CPU(float *a,float *b,float *c,float *d,float *x,int N,int n)
{
	int step,iEven,iLeft,iRight,Delta;     //step number, (mathematically)even index, indices shifted by Delta
	float k1,k2;                            
	auto start = std::chrono::high_resolution_clock::now();

/*------------------------------------------------------------------------------------------------------------*/
	/*Forward Reduction of Initial Matrix*/
/*------------------------------------------------------------------------------------------------------------*/
	for(step = 0 ; step <n-1; step++){
		for(iEven = int_pow(2.0,step+1)-1;iEven < N;iEven = iEven + int_pow(2.0,step+1)){
			Delta = int_pow(2.0,step);
			iLeft = iEven- Delta;
			iRight = iEven+ Delta;

            k1 = a[iEven] / b[iLeft];
            if(iRight>=N){
                iRight = 0;                //some dummy index
                k2 = 0;
            }
            else{
                k2 = c[iEven] / b[iRight];
            }
            a[iEven] = -k1* a[iLeft];
            b[iEven] = b[iEven] - k1 *c[iLeft]  - k2* a[iRight];
            c[iEven] = -k2* c[iRight];
            d[iEven] = d[iEven] -  k1* d[iLeft] - k2* d[iRight];
		}

        
	}

/*------------------------------------------------------------------------------------------------------------*/
	/*Solve the Remaining 2 Equations*/
/*------------------------------------------------------------------------------------------------------------*/
    int iMiddle = (N)/2-1;
    int iFinal = (N-1);

    x[iMiddle] = (d[iMiddle] - c[iMiddle]*d[iFinal]/b[iFinal])/(b[iMiddle]-c[iMiddle]*a[iFinal]/b[iFinal]);
	x[iFinal] = (d[iFinal]-a[iFinal]*d[iMiddle]/b[iMiddle])/(b[iFinal]-a[iFinal]*c[iMiddle]/b[iMiddle]);
/*------------------------------------------------------------------------------------------------------------*/
	/*Backward Substitution*/
/*------------------------------------------------------------------------------------------------------------*/
    int iKnown;
	for(step = n-2; step >= 0;step--){
		for(iKnown=int_pow(2.0,step+1)-1;iKnown<N;iKnown=iKnown+int_pow(2.0,step+1)){
			Delta = int_pow(2.0,step);
			iLeft = iKnown-Delta;
			iRight = iKnown+Delta;
            
            if(iRight>=N){
            break;}

			x[iLeft] = (d[iLeft] - a[iLeft]*x[iLeft-Delta] - c[iLeft]*x[iLeft+Delta])/b[iLeft];  
			x[iRight] = (d[iRight] - a[iRight]*x[iRight-Delta] - c[iRight]*x[iRight+Delta])/b[iRight]; 

		}
	}
	auto stop = std::chrono::high_resolution_clock::now();
	//auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
	float t_cpu = duration.count();
    return t_cpu;
}



//set up matrix from discretized Schrödinger equation
void MATRIX(float **a,float **b,float **c,float **d,float **x,int N)
{
	int i;
	*a = (float *) malloc(sizeof(float)*N);
	*b = (float *) malloc(sizeof(float)*N);
	*c = (float *) malloc(sizeof(float)*N);
	*d = (float *) malloc(sizeof(float)*N);
	*x = (float *) malloc(sizeof(float)*N);

	for(i=0; i<N ;i++){
		(*a)[i] = 1.0;
		(*b)[i] =  -2.0;
		(*c)[i] = 1.0;
        (*x)[i] = 0.0;
        (*d)[i] = exp(-pow(i,2.0));//Gaussian profile
	}
	(*a)[0] = 0.0;
	(*c)[N-1]=0.0;
}


//calculate the block size according to n_eqns
void SET_DIMENSIONS(int n_eqns,int N, dim3 *block, dim3 *grid) 
{
    int block_size;

    if (log2f(n_eqns) >= 10) {
        block_size = 32;
    } else {
        block_size = pow(4, floor(log2f(n_eqns) / 2) / 2);  // Same logic as second version
    }

    // Use a 1D block with size block_size * block_size
    block->x = block_size * block_size;

    // Compute how many blocks are needed to cover n_eqns
    int threads_per_block = block->x;
    grid->x = (n_eqns + threads_per_block - 1) / threads_per_block;
}

//convert array to .csv file
void TO_CSV(float *x_gpu, float *x_cpu, int N) {
    FILE *fp;
    char filename[] = "x_gpu.csv";
    fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("Error opening x_gpu.csv");
        return; 
    }
    fclose(fp);
    std:: cout<< "GPU array data exported to "<< filename<<std::endl;

    FILE *fp1;
    char filename1[] = "x_cpu.csv";
    fp1 = fopen(filename1, "w");
    if (fp1 == NULL) {
        perror("Error opening x_cpu.csv");
        return; 
    }

    fclose(fp1);
    std::cout<<"CPU array data exported to "<< filename1 <<std::endl;
}


//calculate max(|x_gpu-x_cpu|)
void find_max_difference(float *x, float *x_cpu, int N) 
{
    float max_diff = 0.0;
    int max_index = 0;
    for (int i = 0; i < N; i++) {
        float diff = fabs(x[i]-x_cpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_index = i;
        }
    }
    std::cout<<"\nLargest error found at i ="<< max_index<< ", where x_gpu-x_cpu=" << x[max_index] << "-" << x_cpu[max_index] <<"="<< max_diff <<std::endl;
}

//CUDA Check macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
		std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;\
		exit(-1); \
    } \
}
