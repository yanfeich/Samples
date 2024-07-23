#include <time.h>
#include <unistd.h>
#include <string.h>
#include <algorithm>
#include <synapse_api.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <map>

int main(int argc, char *argv[])
{
    unsigned M = 256;
    unsigned N = 1024;
    unsigned K = 2048;

    std::map<synDataType, unsigned> dtype2size{
        {syn_type_fp8_143, 1},
        {syn_type_fp8_152, 1},
        {syn_type_bf16, 2},
        {syn_type_fp16, 2},
        {syn_type_uint32, 4},
        {syn_type_int32, 4},
        {syn_type_single, 4},
    };

    synStatus status = synInitialize();
    assert(status == synSuccess && "Failed to call  synInitialize()");

    uint32_t deviceId = 0;
    status = synDeviceAcquire(&deviceId, nullptr);
    assert(status == synSuccess && "Failed to call  synDeviceAcquire()");

    synDeviceInfo deviceInfo;
    status = synDeviceGetInfo(deviceId, &deviceInfo);
    assert(status == synSuccess && "Failed to call  synDeviceGetInfo()");

    synDeviceType device_type = deviceInfo.deviceType;

    // ************************************************ compile
    synGraphHandle graphHandle = nullptr;
    status = synGraphCreate(&graphHandle, device_type);
    assert(status == synSuccess && "Failed to call synGraphCreate()");

    std::vector<synTensor> inputs;
    std::vector<synTensor> outputs;

    // ************************************************ create A tensor
    unsigned int offset = 0;
    unsigned A_shape[] = {K, M};
    unsigned A_size = K * M * 2;
    synSectionHandle A_SectionHandle = nullptr;
    status = synSectionCreate(&A_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor A_desc;
    A_desc.m_dataType = syn_type_bf16;
    A_desc.m_dims = 2UL;
    A_desc.m_name = "A";
    memset(A_desc.m_strides, 0, sizeof(A_desc.m_strides));
    memset(A_desc.m_sizes, 0, sizeof(A_desc.m_sizes));
    memcpy(A_desc.m_sizes, A_shape, 2 * sizeof(unsigned));

    synTensor syn_A = nullptr;
    status = synTensorCreate(&syn_A, &A_desc, A_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    inputs.push_back(syn_A);

    //************************************************ create B tensor
    unsigned B_shape[] = {N, K};
    unsigned B_size = N * K * 2;
    synSectionHandle B_SectionHandle = nullptr;
    status = synSectionCreate(&B_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor B_desc;
    B_desc.m_dataType = syn_type_bf16;
    B_desc.m_dims = 2UL;
    B_desc.m_name = "B";
    memset(B_desc.m_strides, 0, sizeof(B_desc.m_strides));
    memset(B_desc.m_sizes, 0, sizeof(B_desc.m_sizes));
    memcpy(B_desc.m_sizes, B_shape, 2 * sizeof(unsigned));
    synTensor syn_B = nullptr;
    status = synTensorCreate(&syn_B, &B_desc, B_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    inputs.push_back(syn_B);

    //************************************************ create C tensor
    // NCHW => WHCN
    unsigned C_shape[] = {N, M};
    unsigned C_size = N * M * 2;
    synSectionHandle C_SectionHandle = nullptr;
    status = synSectionCreate(&C_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor C_desc;
    C_desc.m_dataType = syn_type_bf16;
    C_desc.m_dims = 2UL;
    C_desc.m_name = "C";
    memset(C_desc.m_strides, 0, sizeof(C_desc.m_strides));
    memset(C_desc.m_sizes, 0, sizeof(C_desc.m_sizes));
    memcpy(C_desc.m_sizes, C_shape, 2 * sizeof(unsigned));
    synTensor syn_C = nullptr;

    status = synTensorCreate(&syn_C, &C_desc, C_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    outputs.push_back(syn_C);

    //************************************************ create gemm node
    std::string guid = "gemm";
    std::string node_name = "GEMM";
    synGEMMParams gemm_params;
    bool transpose_a = false;
    bool transpose_b = false;
    gemm_params.transpose_a = transpose_a;
    gemm_params.transpose_b = transpose_b;

    status = synNodeCreate(graphHandle,
                           inputs.data(),
                           outputs.data(),
                           2,
                           1,
                           &gemm_params,
                           sizeof(gemm_params),
                           guid.c_str(),
                           node_name.c_str(),
                           nullptr,
                           nullptr);
    assert(status == synSuccess && "Failed to call synNodeCreate()");

    //************************************************ compile graph
    synRecipeHandle recipeHandle = nullptr;
    std::string name("gemm.recipe");
    status = synGraphCompile(&recipeHandle,
                             graphHandle,
                             name.c_str(),
                             nullptr);
    assert(status == synSuccess && "Failed to call synGraphCompile()");

    // ************************************************ runtime
    uint64_t workspace = 0;
    status = synWorkspaceGetSize(&workspace, recipeHandle);
    assert(status == synSuccess && "Failed to call synWorkspaceGetSize()");

    uint64_t hbm_addr = 0;
    status = synDeviceMalloc(deviceId, workspace, 0, 0, &hbm_addr);
    assert(status == synSuccess && "Failed to call synDeviceMalloc()");

    std::vector<void *> host_inputs;
    std::vector<void *> host_outputs;

    // malloc host A tensor
    uint64_t A_device = 0;
    status = synDeviceMalloc(deviceId, A_size, 0, 0, &A_device);
    assert(status == synSuccess && "Failed to call A synDeviceMalloc()");

    // malloc device A tensor
    unsigned char *A_host = nullptr;
    status = synHostMalloc(deviceId, A_size, 0, (void **)&A_host);
    assert(status == synSuccess && "Failed to call A synHostMalloc()");
    memset(A_host, 0, A_size);
    host_inputs.push_back(A_host);

    // malloc host B tensor
    unsigned char *B_host = nullptr;
    status = synHostMalloc(deviceId, B_size, 0, (void **)&B_host);
    assert(status == synSuccess && "Failed to call B synHostMalloc()");
    memset(B_host, 0, B_size);
    host_inputs.push_back(B_host);

    // malloc device B tensor
    uint64_t B_device = 0;
    status = synDeviceMalloc(deviceId, B_size, 0, 0, &B_device);
    assert(status == synSuccess && "Failed to call B synDeviceMalloc()");

    // malloc device C tensor
    uint64_t C_device = 0;
    status = synDeviceMalloc(deviceId, C_size, 0, 0, &C_device);
    assert(status == synSuccess && "Failed to call C synDeviceMalloc()");

    // malloc host C tensor
    unsigned char *C_host = nullptr;
    status = synHostMalloc(deviceId, C_size, 0, (void **)&C_host);
    assert(status == synSuccess && "Failed to call C synHostMalloc()");
    memset(C_host, 0, C_size);
    host_outputs.push_back(C_host);

    synStreamHandle h2d, d2h, compute;
    status = synStreamCreateGeneric(&h2d, deviceId, 0);
    assert(status == synSuccess && "Failed to create h2d");

    status = synStreamCreateGeneric(&d2h, deviceId, 0);
    assert(status == synSuccess && "Failed to create d2h");

    status = synStreamCreateGeneric(&compute, deviceId, 0);
    assert(status == synSuccess && "Failed to create compute");

    synEventHandle h2d_event = nullptr;
    status = synEventCreate(&h2d_event, deviceId, 0);
    assert(status == synSuccess && "Failed to call synEventCreate()");

    status = synMemCopyAsync(h2d, (uint64_t)A_host, A_size, A_device, HOST_TO_DRAM);
    assert(status == synSuccess && "Failed to call A synMemCopyAsync()");

    status = synMemCopyAsync(h2d, (uint64_t)B_host, B_size, B_device, HOST_TO_DRAM);
    assert(status == synSuccess && "Failed to call B synMemCopyAsync()");

    status = synMemCopyAsync(h2d, (uint64_t)C_host, C_size, C_device, HOST_TO_DRAM);
    assert(status == synSuccess && "Failed to call C synMemCopyAsync()");

    status = synEventRecord(h2d_event, h2d);
    assert(status == synSuccess && "Failed to call synEventRecord()");

    status = synEventQuery(h2d_event);
    std::cout << "h2d_event status: " << status << std::endl;

    status = synStreamWaitEvent(compute, h2d_event, 0);
    assert(status == synSuccess && "Failed to call synEventRecord()");
    // std::vector<synLaunchTensorInfoExt> *graph_inputs = nullptr;
    // std::vector<synLaunchTensorInfoExt> *graph_outputs = nullptr;
    synLaunchTensorInfoExt A_launch;
    A_launch.tensorName = "A";
    A_launch.pTensorAddress = A_device;
    synLaunchTensorInfoExt B_launch;
    B_launch.tensorName = "B";
    B_launch.pTensorAddress = B_device;
    synLaunchTensorInfoExt C_launch;
    C_launch.tensorName = "C";
    C_launch.pTensorAddress = C_device;

    std::vector<synLaunchTensorInfoExt> launch_info = {A_launch, B_launch, C_launch};

    status = synLaunchExt(compute, launch_info.data(), launch_info.size(), hbm_addr, recipeHandle, 0);

    status = synStreamSynchronize(compute);
    assert(status == synSuccess && "Failed to call synStreamSynchronize()");

    // copy data from device to host
    status = synMemCopyAsync(d2h, C_device, C_size, (uint64_t)C_host, DRAM_TO_HOST);
    std::cout << "C " << status << std::endl;
    assert(status == synSuccess && "Failed to call C synMemCopyAsync()");

    status = synStreamSynchronize(d2h);
    assert(status == synSuccess && "Failed to call synStreamSynchronize()");

    // query device status
    status = synEventQuery(h2d_event);
    std::cout << "h2d_event status: " << status;

    status = synStreamQuery(compute);
    std::cout << "compute stream status: " << status << std::endl;

    // clean
    status = synEventDestroy(h2d_event);
    assert(status == synSuccess && "Failed to call synEventDestroy()");

    status = synGraphDestroy(graphHandle);
    assert(status == synSuccess && "Failed to call synGraphDestroy()");

    status = synRecipeDestroy(recipeHandle);
    assert(status == synSuccess && "Failed to call synRecipeDestroy()");

    status = synHostFree(deviceId, A_host, 0);
    assert(status == synSuccess && "Failed to call synHostFree()");

    status = synHostFree(deviceId, B_host, 0);
    assert(status == synSuccess && "Failed to call synHostFree()");

    status = synHostFree(deviceId, C_host, 0);
    assert(status == synSuccess && "Failed to call synHostFree()");

    status = synStreamDestroy(h2d);
    assert(status == synSuccess && "Failed to call synStreamDestroy()");

    status = synStreamDestroy(d2h);
    assert(status == synSuccess && "Failed to call synStreamDestroy()");

    status = synStreamDestroy(compute);
    assert(status == synSuccess && "Failed to call synStreamDestroy()");

    status = synDeviceRelease(deviceId);
    assert(status == synSuccess && "Failed to call synDeviceRelease()");

    status = synDestroy();
    assert(status == synSuccess && "Failed to call synDestroy()");
    return 0;
}
