/*
This program implements a tridiagonal solver following the Cyclic Reduction Method.
Requires functions defined in the file "CR.cpp"
*/
#include "CR.cpp"
#include<iostream>
#include<iomanip> //for printing precision
#include<chrono>


void CR_GPU(float *a, float *b, float *c, float *d, float *x_gpu,int N,int n){


	int step;               	 	//keeps track of current step during forward/backward stage
	int n_eqns;             	 	//number of equations to be modified per step
	dim3 dimBlock,dimGrid;  	    //stores GPU block/grid dimensions

	float *dev_a; 			    //lower diagonal vector
	float *dev_b; 			    //main diagonal vector
	float *dev_c; 			    //upper diagonal vector
	float *dev_d;	    	    //RHS of equations
	float *dev_x_gpu; //vector of unknowns
	dev_a = dev_b = dev_c = dev_x_gpu = dev_d=NULL;
	cudaFree(0);
	auto start = std::chrono::high_resolution_clock::now(); //record current time to clock GPU process 

	//allocate memory on GPU 
	CUDA_CHECK(cudaMalloc(&dev_a,sizeof(float)*N));
	CUDA_CHECK(cudaMalloc(&dev_b,sizeof(float)*N));
	CUDA_CHECK(cudaMalloc(&dev_c,sizeof(float)*N));
	CUDA_CHECK(cudaMalloc(&dev_x_gpu,sizeof(float)*N));
	CUDA_CHECK(cudaMalloc(&dev_d,sizeof(float)*N));

	//copy arrays to GPU
	CUDA_CHECK(cudaMemcpy(dev_a,a,sizeof(float)*N,cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_b,b,sizeof(float)*N,cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_c,c,sizeof(float)*N,cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_d,d,sizeof(float)*N,cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_x_gpu,x_gpu,sizeof(float)*N,cudaMemcpyHostToDevice));

	cudaFuncSetCacheConfig(FORWARD_GPU,cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(BACKWARD_GPU,cudaFuncCachePreferL1);



	//Forward Reduction 
	for(step=0;step<n-1;step++){
		n_eqns = (N-int_pow(2.0,step+1))/int_pow(2.0,step+1)+1;
		SET_DIMENSIONS(n_eqns,N,&dimBlock,&dimGrid);
		FORWARD_GPU<<<dimGrid,dimBlock>>>(dev_a, dev_b, dev_c, dev_d, dev_x_gpu,N,n_eqns,step);
	}
	CUDA_CHECK(cudaDeviceSynchronize());
	//Solve remaining 2 equations
	SOLVE_EQ_GPU<<<1,1>>>(dev_a, dev_b, dev_c, dev_d, dev_x_gpu, (N) / 2 - 1, N - 1);  
	CUDA_CHECK(cudaDeviceSynchronize());

	//Backward Substitution
	for(step=n-2;step>=0;step--){
		n_eqns = (N-int_pow(2.0,step+1))/int_pow(2.0,step+1)+1;
		SET_DIMENSIONS(n_eqns,N,&dimBlock,&dimGrid);
		BACKWARD_GPU<<<dimGrid,dimBlock>>>(dev_a, dev_b, dev_c, dev_d, dev_x_gpu,N,n_eqns,step);
	}

	//copy gpu array back to host
	CUDA_CHECK(cudaMemcpy(x_gpu,dev_x_gpu,sizeof(float)*N,cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaDeviceSynchronize());
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono:: duration<float>> (stop-start);
	float t_gpu = duration.count();
	cudaDeviceReset();

	std::cout <<"\nCompute Time on GPU ="<< std::fixed<<std::setprecision(4)<<t_gpu;

}


int main(){

	int n = 24;            
	int N = int_pow(2,n);        	//length of main diagonal
	float *a; 			    //lower diagonal vector
	float *b; 			    //main diagonal vector
	float *c; 			    //upper diagonal vector
	float *d;	    	    //RHS of equations
	float *x_gpu,*x_cpu; //vector of unknowns

	a = b = c = x_gpu = d = NULL;


	

	MATRIX(&a,&b,&c,&d,&x_gpu,N);	//set values for a,b,c and d

	

	
/*-------------------------------------------Execution on GPU-------------------------------------------*/
	CR_GPU(a,b,c,d,x_gpu,N, n);
	
/*-------------------------------------------Execution on CPU-------------------------------------------*/
	x_cpu  = new float[N]();
	float t_cpu; 
	t_cpu = CR_CPU(a,b,c,d,x_cpu,N,n);
	std:: cout<<"\nCompute Time on CPU ="<<std::fixed<<std::setprecision(4)<<t_cpu;
	find_max_difference(x_gpu, x_cpu, N);

    /*
	std::cout<< "GPU Array elements: "<< std::endl;
    for (int i = 0; i < N; i++) {
        std::cout<< x_gpu[i];
    }
    std::cout<< std::endl;
      
	std::cout<< "CPU Array elements: "<< std::endl;
    for (int i = 0; i < N; i++) {
        std::cout<< x_cpu[i];
    }
    std::cout<< std::endl;
	*/

	//TO_CSV(x_gpu,x_cpu,N);

	return 0;

}
