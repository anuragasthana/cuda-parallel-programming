#include "memory_analysis.h"

__global__ void globalMemorySearch(int *input, int totalFound, int numElements, int numThreads)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId < numThreads)
    {
        int threadSpan = numElements / numThreads;
        int offset = threadId * threadSpan;

        for (int i = offset; i < offset + threadSpan; ++i)
        {
            input[i]++; // Increment before comparison
            if (input[i] == constant_search_value)
            {
                input[i] = 1;
            }
        }
    }
}

__global__ void sharedMemorySearch(int *input, int totalFound, int numElements, int numThreads)
{
    extern __shared__ int sharedInput[];
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int threadSpan = numElements / numThreads;
    int offset = threadId * threadSpan;

    if (threadId < numThreads)
    {
        for (int i = offset; i < offset + threadSpan; ++i)
        {
            sharedInput[i] = input[i]; // Copy to shared memory
        }
        __syncthreads(); // Synchronize threads to ensure all data is in shared memory

        for (int i = offset; i < offset + threadSpan; ++i)
        {
            sharedInput[i]++;
            if (sharedInput[i] == constant_search_value)
            {
                sharedInput[i] = 1;
            }
        }
        __syncthreads(); // Synchronize before writing back to global memory

        for (int i = offset; i < offset + threadSpan; ++i)
        {
            input[i] = sharedInput[i]; // Write back to global memory
        }
    }
}

__global__ void constantMemorySearch(int totalFound)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId < constant_num_threads)
    {
        int offset = threadId * constant_thread_span;

        for (int i = offset; i < offset + constant_thread_span; ++i)
        {
            constant_input[i]++; // Increment the value in constant memory
            if (constant_input[i] == constant_search_value)
            {
                constant_input[i] = 1; // Modify constant memory
            }
        }
    }
}

__global__ void registerMemorySearch(int *input, int totalFound, int numElements, int numThreads)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId < numThreads)
    {
        int threadSpan = numElements / numThreads;
        int offset = threadId * threadSpan;
        int register_search_value = constant_search_value; // Copy to register

        for (int i = offset; i < offset + threadSpan; ++i)
        {
            input[i]++;
            if (input[i] == register_search_value)
            {
                input[i] = 1;
            }
        }
    }
}

// This will generate an array of size numElements of random integers from 0 to 255 in pageable host memory
__host__ int* allocatePageableRandomHostMemory(int numElements)
{
    srand(time(0));
    size_t size = numElements * sizeof(int);

    // Allocate the host pinned memory input pointer B
    int *data;
    cudaHostAlloc((void**)&data, size, cudaHostAllocDefault);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        data[i] = rand() % 255;
    }

    return data;
}

__host__ int * allocateDeviceMemory(int numElements)
{
    size_t size = numElements * sizeof(int);

    int *d_input = NULL;
    cudaError_t err = cudaMalloc(&d_input, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return d_input;
}

__host__ void copyFromHostToDevice(std::string kernelType, int *input, int numElements, int numThreads, int *d_input)
{
    size_t size = numElements * sizeof(int);
    cudaError_t err;

    if (!strcmp(kernelType.c_str(), "constant"))
    {
        // constant_num_threads = numThreads;
        cudaError_t err = cudaMemcpyToSymbol(constant_num_threads, &numThreads, sizeof(int), 0, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy num threads from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // constant_num_elements = numElements; // Copy numElements to constant memory
        err = cudaMemcpyToSymbol(constant_num_elements, &numElements, sizeof(int), 0, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy num elements from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy the input data to constant memory
        size_t constant_size = numElements * sizeof(int);
        err = cudaMemcpy(constant_input, input, constant_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy array input from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        int threadSpan = numElements / numThreads;
        err = cudaMemcpyToSymbol(constant_thread_span, &threadSpan, sizeof(int), 0, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy thread span from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    err = cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy array input from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    
    // size_t size = numElements * sizeof(int);

    // if(!strcmp(kernelType.c_str(), "constant"))
    // {
    //     // Copy input, numElements, threadSpan, and numThreads to constant memory
    //     int threadSpan = numElements / numThreads;
    // } else
    // {
    //     // Copy input to d_input
    //     if (err != cudaSuccess)
    //     {
    //         fprintf(stderr, "Failed to copy array input from host to device (error code %s)!\n", cudaGetErrorString(err));
    //         exit(EXIT_FAILURE);
    //     }
    // }
}

__host__ void executeKernel(int *d_input, int numElements, int threadsPerBlock, std::string kernelType)
{
    int totalFound = 0;
    // Launch the search CUDA Kernel
    if (!strcmp(kernelType.c_str(), "global"))
    {
        globalMemorySearch<<<1, threadsPerBlock>>>(d_input, totalFound, numElements, threadsPerBlock); // you will need to fill in function arguments appropriately
    } else if (!strcmp(kernelType.c_str(), "constant"))
    {
        constantMemorySearch<<<1, threadsPerBlock>>>(totalFound); // you will need to fill in function arguments appropriately
    } else if (!strcmp(kernelType.c_str(), "shared"))
    {
        unsigned int_array_size = numElements * sizeof(int);
        sharedMemorySearch<<<1, threadsPerBlock, int_array_size>>>(d_input, totalFound, numElements, threadsPerBlock); // you will need to fill in function arguments appropriately
    } else {
        registerMemorySearch<<<1, threadsPerBlock>>>(d_input, totalFound, numElements, threadsPerBlock); // you will need to fill in function arguments appropriately
    }
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch %s kernel (error code %s)!\n", kernelType.c_str(), cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();
}

// Free device global memory
__host__ void deallocateMemory(int *d_input)
{

    cudaError_t err = cudaFree(d_input);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

// Reset the device and exit
__host__ void cleanUpDevice()
{
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


__host__ std::tuple<int, std::string, int, std::string> parseCommandLineArguments(int argc, char *argv[])
{
    int elementsPerThread = 2;
    int threadsPerBlock = 256;
    std::string currentPartId = "test";
    std::string kernelType = "global";

    for(int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if(option.compare("-t") == 0) 
        {
            threadsPerBlock = atoi(value.c_str());
        }
        else if(option.compare("-m") == 0) 
        {
            elementsPerThread = atoi(value.c_str());
        }
        else if(option.compare("-p") == 0) 
        {
            currentPartId = value;
        }
        else if(option.compare("-k") == 0) 
        {
            kernelType = value;
        }
    }

    return {elementsPerThread, currentPartId, threadsPerBlock, kernelType};
}

__host__ int * setUpInput(int numElements)
{
    srand(time(0));
    int *input;

    int searchValue = rand() % 255;
    cudaError_t err = cudaMemcpyToSymbol(constant_search_value, &searchValue, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy constant int d_v from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    input = allocatePageableRandomHostMemory(numElements);

    return input;
}

/*
 * Host main routine
 * -m elementsPerThread - the number of elements that a thread will search for a random value in
 * -p currentPartId - the Coursera Part ID
 * -t threadsPerBlock - the number of threads to schedule for concurrent processing
 * -k the kernel type - global, constant, shared, register
 */
int main(int argc, char *argv[])
{
    auto[elementsPerThread, currentPartId, threadsPerBlock, kernelType] = parseCommandLineArguments(argc, argv);

    int numElements = elementsPerThread * threadsPerBlock;
    if (numElements > 16384)
    {
        elementsPerThread = 16384 / threadsPerBlock;
        numElements = elementsPerThread * threadsPerBlock;
        std::cout << "Adjusted elements per thread to " << elementsPerThread << " to keep total elements under 16K." << std::endl;
    }

    int *input = setUpInput(numElements);
    int *d_input = allocateDeviceMemory(numElements);

    copyFromHostToDevice(kernelType, input, numElements, threadsPerBlock, d_input);

    // Start time including kernel processing time
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    executeKernel(d_input, numElements, threadsPerBlock, kernelType);

    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start,stop);

    deallocateMemory(d_input);
    cleanUpDevice();
    
    ofstream outputfile;
    outputfile.open ("output.csv", std::ios_base::app);
    outputfile << currentPartId.c_str() << "," << kernelType.c_str() << "," << threadsPerBlock << "," << elementsPerThread << "," << elapsedTime << "\n";
    outputfile.close();

    return 0;
}