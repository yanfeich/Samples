#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <chrono>
#include <map>
#include <numeric>
#include <algorithm>
#include "synapse_api.h"
#include <dlfcn.h>
#include <vector>

#define ERRCODE_SUCCESS                0
#define ERRCODE_MISSING_ARGUMENTS      1
#define ERRCODE_DLOPEN_FAILED          2
#define ERRCODE_SYNAPSE_FAILED         3
#define ERRCODE_TENSOR_INPUT_FAILED    4


#define CHECK_STATUS(status, stage) \
            if (status != synSuccess) \
            { \
                printf("%s failed!\n", stage); \
                exit(ERRCODE_SYNAPSE_FAILED); \
            }

int main(int argc, char *argv[])
{
    int numOfIters = 100000;
    uint32_t size = 10 * 1024 * 1024; // 10MB
    
    synStatus status;

    status = synInitialize();
    CHECK_STATUS(status, "synInitialize : Synapse initialization failed");

    synDeviceId deviceId(0);
    std::cout << "Acquiring Device...\n";
    status = synDeviceAcquireByDeviceType(&deviceId, synDeviceGaudi2);
    CHECK_STATUS(status, "Failed to acquire device!");
    std::cout << "Acquired Device ID: " << deviceId << "\n";

    // Host buffers for input & output
    unsigned char* inputsBuffers;
    unsigned char* outputsBuffers;

    std::cout << "Allocating inputs on host: size: " << size << std::endl;
    status = synHostMalloc(deviceId, size , 0, (void **)&inputsBuffers);
    CHECK_STATUS(status, "Could not allocate host memory for inputs/outputs");
    // Init input with random values and zero-out the output
    std::srand(42);
    for (int j = 0; j < size; j++)
    {
        inputsBuffers[j] = (unsigned char)std::rand();
    }

    uint64_t workspaceSize = size;
    uint64_t workspaceAddr(0);
    uint64_t d2d_output(0);
    std::cout << "Allocating workspace Size: " << workspaceSize << " on the device...\n";
    status = synDeviceMalloc(deviceId, workspaceSize, 0, 0, &workspaceAddr);
    CHECK_STATUS(status, "Memory allocation for workspace failed!");

    status = synDeviceMalloc(deviceId, workspaceSize, 0, 0, &d2d_output);
    CHECK_STATUS(status, "Memory allocation for workspace failed!");

    // Create streams
    synStreamHandle copyInStream;
    synStreamHandle copyOutStream;
    synStreamHandle copyd2dStream;

    status = synStreamCreateGeneric(&copyInStream, deviceId, 0);
    CHECK_STATUS(status, "create stream to copy data to the device failed");
    
    status = synStreamCreateGeneric(&copyOutStream, deviceId, 0);
    CHECK_STATUS(status, "create stream to copy data from the device failed");

    status = synStreamCreateGeneric(&copyd2dStream, deviceId, 0);
    CHECK_STATUS(status, "create stream to copy data from the device to device failed");

    synEventHandle event_h2d;
    status = synEventCreate(&event_h2d, deviceId, 0);
    CHECK_STATUS(status, "create event failed");

    std::cout << "Starting execution...\n";

    // Copy data (inputs) from host to device
    std::cout << "Starting copy data host to device...\n";
    auto start_full_flow = std::chrono::high_resolution_clock::now();
    for (int i=0; i<numOfIters; i++)
    {
        status = synMemCopyAsync(copyInStream, (uint64_t)inputsBuffers, workspaceSize, workspaceAddr, HOST_TO_DRAM);
        CHECK_STATUS(status, "copy inputs to device memory failed");
    }

    // Associate an event with its completion
    status = synEventRecord(event_h2d, copyInStream);
    CHECK_STATUS(status, "record event failed");

    // Copy waits for compute to finish
    status = synStreamWaitEvent(copyInStream, event_h2d, 0);
    CHECK_STATUS(status,"stream wait event");
    auto end_h2d = std::chrono::high_resolution_clock::now();

    /****************************************************************/
    std::cout << "Allocating inputs on host: size: " << size << std::endl;
    status = synHostMalloc(deviceId, size , 0, (void **)&outputsBuffers);
    CHECK_STATUS(status, "Could not allocate host memory for inputs/outputs");
    // Init input with random values and zero-out the output
    memset(outputsBuffers,0, size);

    // Copy data from device to host
    std::cout << "Starting copy data device to host...\n";
    auto start_d2h = std::chrono::high_resolution_clock::now();
    for (int i=0; i<numOfIters; i++){
        status = synMemCopyAsync(copyOutStream, workspaceAddr, workspaceSize, (uint64_t)outputsBuffers, DRAM_TO_HOST);
        CHECK_STATUS(status, "copy outputs to host memory");
    }

    // Wait for everything to finish by blocking on the copy from device to host
    status = synStreamSynchronize(copyOutStream);
    CHECK_STATUS(status, "wait for copy out stream");
    auto end_d2h = std::chrono::high_resolution_clock::now();

    std::cout << "testing HBM BW ...\n";
    auto start_d2d = std::chrono::high_resolution_clock::now();
    for (int i=0; i<numOfIters; i++){
        status = synMemCopyAsync(copyd2dStream, workspaceAddr, workspaceSize, d2d_output, DRAM_TO_DRAM);
        CHECK_STATUS(status, "copy d2d");
    }

    // Wait for everything to finish by blocking on the copy from device to host
    status = synStreamSynchronize(copyd2dStream);
    CHECK_STATUS(status, "wait for copy out stream");

    auto end_full_flow = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> h2d_elapsed = end_h2d - start_full_flow;
    std::chrono::duration<double> d2h_elapsed = end_d2h - start_d2h;
    std::chrono::duration<double> d2d_elapsed = end_full_flow - start_d2d;

    std::cout << "Execution completed successfully:\nh2d took "
         << h2d_elapsed.count()
         << " s, throughput: "
         << workspaceSize * numOfIters / h2d_elapsed.count() / 1000000000
         << " GB/s\nd2h took "
         << d2h_elapsed.count()
         << " s, throughput: "
         << workspaceSize * numOfIters / d2h_elapsed.count() / 1000000000
         << " GB/s\nd2d took "
         << d2d_elapsed.count()
         << " s, throughput: "
         << workspaceSize * numOfIters / d2d_elapsed.count() / 1000000000
         << " GB/s\n";

    bool consist = true;
    for (int j = 0; j < size; j++)
    {
        if(inputsBuffers[j] != outputsBuffers[j]){
            consist = false;
            CHECK_STATUS(synFail, "input/output buffer data check ");
        }
    }
    if(consist == true)
        std::cout << "Input / Output Buffer Data consistency test passed.\n";
    
    std::cout << "Freeing all allocations...\n";
    status = synStreamDestroy(copyInStream);
    CHECK_STATUS(status, "Failed to destroy copy in stream!");
    status = synStreamDestroy(copyOutStream);
    CHECK_STATUS(status, "Failed to destroy copy out stream!");
    
    status = synDeviceFree(deviceId, workspaceAddr, 0);
    CHECK_STATUS(status, "Failed to free workspace!");
    status = synDeviceRelease(deviceId);
    CHECK_STATUS(status, "Failed to free device!");
    std::cout << "Freeing succeeded, exiting now.\n";
    status = synDestroy();
    CHECK_STATUS(status, "Failed to destroy synapse");

    return ERRCODE_SUCCESS;
}
