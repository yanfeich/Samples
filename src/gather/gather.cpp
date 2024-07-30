#include <time.h>
#include <unistd.h>
#include <string.h>
#include <algorithm>
#include <synapse_api.h>
#include <synapse_common_types.hpp>
#include <perf_lib_layer_params.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <map>

#define INDEX_DIM_1

int main(int argc, char *argv[])
{
    unsigned L = 500;
    unsigned H = 4096;
    unsigned B = 64;
    unsigned M = 20;

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
    assert(status == synSuccess && "Failed to call synInitialize()");

    uint32_t deviceId = 0;
    status = synDeviceAcquire(&deviceId, nullptr);
    assert(status == synSuccess && "Failed to call synDeviceAcquire()");

    synDeviceInfo deviceInfo;
    status = synDeviceGetInfo(deviceId, &deviceInfo);
    assert(status == synSuccess && "Failed to call synDeviceGetInfo()");

    synDeviceType device_type = deviceInfo.deviceType;

    // ************************************************ compile
    synGraphHandle graphHandle = nullptr;
    status = synGraphCreate(&graphHandle, device_type);
    assert(status == synSuccess && "Failed to call synGraphCreate()");

    std::vector<synTensor> inputs;
    std::vector<synTensor> outputs;

    // ************************************************ create X tensor
    unsigned int offset = 0;
    unsigned X_shape[] = {L, H};
    unsigned X_size = L * H * 2;
    synSectionHandle X_SectionHandle = nullptr;
    status = synSectionCreate(&X_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor X_desc;
    X_desc.m_dataType = syn_type_bf16;
    X_desc.m_dims = 2UL;
    X_desc.m_name = "X";
    memset(X_desc.m_strides, 0, sizeof(X_desc.m_strides));
    memset(X_desc.m_sizes, 0, sizeof(X_desc.m_sizes));
    memcpy(X_desc.m_sizes, X_shape, 2 * sizeof(unsigned));

    synTensor syn_X = nullptr;
    status = synTensorCreate(&syn_X, &X_desc, X_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    inputs.push_back(syn_X);

#ifdef INDEX_DIM_1
    // ************************************************ create I tensor
    unsigned I_shape[] = {1};
    unsigned I_size = 1 * 4;
    synSectionHandle I_SectionHandle = nullptr;
    status = synSectionCreate(&I_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor I_desc;
    I_desc.m_dataType = syn_type_int32;
    I_desc.m_dims = 1UL;
    I_desc.m_name = "I";
    memset(I_desc.m_strides, 0, sizeof(I_desc.m_strides));
    memset(I_desc.m_sizes, 0, sizeof(I_desc.m_sizes));
    memcpy(I_desc.m_sizes, I_shape, 1 * sizeof(unsigned));

    synTensor syn_I = nullptr;
    status = synTensorCreate(&syn_I, &I_desc, I_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    inputs.push_back(syn_I);
    
    //************************************************ create Y tensor
    unsigned Y_shape[] = {1, H};
    unsigned Y_size = 1 * H * 2;
    synSectionHandle Y_SectionHandle = nullptr;
    status = synSectionCreate(&Y_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor Y_desc;
    Y_desc.m_dataType = syn_type_bf16;
    Y_desc.m_dims = 2UL;
    Y_desc.m_name = "Y";
    memset(Y_desc.m_strides, 0, sizeof(Y_desc.m_strides));
    memset(Y_desc.m_sizes, 0, sizeof(Y_desc.m_sizes));
    memcpy(Y_desc.m_sizes, Y_shape, 2 * sizeof(unsigned));
    synTensor syn_Y = nullptr;

    status = synTensorCreate(&syn_Y, &Y_desc, Y_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    outputs.push_back(syn_Y);

    //************************************************ create gather node
    std::string guid = "gather_bf16";
    std::string node_name = "Gather_bf16_1";

    ns_GatherKernel::Params params;
    params.axis = 0;
    
    status = synNodeCreate(graphHandle,
                           inputs.data(),
                           outputs.data(),
                           2,
                           1,
                           &params,
                           sizeof(params),
                           guid.c_str(),
                           node_name.c_str(),
                           nullptr,
                           nullptr);
    assert(status == synSuccess && "Failed to call synNodeCreate()");

    //************************************************ compile graph
    synRecipeHandle recipeHandle = nullptr;
    std::string name("gather_1.recipe");
    status = synGraphCompile(&recipeHandle,
                             graphHandle,
                             name.c_str(),
                             nullptr);
    assert(status == synSuccess && "Failed to call synGraphCompile()");
#endif
    /*************************************************************************/

#ifdef INDEX_DIM_2
    // ************************************************ create I tensor
    unsigned I_shape[] = {B, M};
    unsigned I_size = B * M * 4;
    synSectionHandle I_SectionHandle = nullptr;
    status = synSectionCreate(&I_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor I_desc;
    I_desc.m_dataType = syn_type_int32;
    I_desc.m_dims = 2UL;
    I_desc.m_name = "I";
    memset(I_desc.m_strides, 0, sizeof(I_desc.m_strides));
    memset(I_desc.m_sizes, 0, sizeof(I_desc.m_sizes));
    memcpy(I_desc.m_sizes, I_shape, 2 * sizeof(unsigned));

    synTensor syn_I = nullptr;
    status = synTensorCreate(&syn_I, &I_desc, I_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    inputs.push_back(syn_I);
    
    //************************************************ create Y tensor
    unsigned Y_shape[] = {B, M, H};
    unsigned Y_size = B * M * H * 2;
    synSectionHandle Y_SectionHandle = nullptr;
    status = synSectionCreate(&Y_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor Y_desc;
    Y_desc.m_dataType = syn_type_bf16;
    Y_desc.m_dims = 3UL;
    Y_desc.m_name = "Y";
    memset(Y_desc.m_strides, 0, sizeof(Y_desc.m_strides));
    memset(Y_desc.m_sizes, 0, sizeof(Y_desc.m_sizes));
    memcpy(Y_desc.m_sizes, Y_shape, 3 * sizeof(unsigned));
    synTensor syn_Y = nullptr;

    status = synTensorCreate(&syn_Y, &Y_desc, Y_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    outputs.push_back(syn_Y);

    //************************************************ create gather node
    std::string guid = "gather_bf16";
    std::string node_name = "Gather_bf16_1";

    ns_GatherKernel::Params params;
    params.axis = 0;
    
    status = synNodeCreate(graphHandle,
                           inputs.data(),
                           outputs.data(),
                           2,
                           1,
                           &params,
                           sizeof(params),
                           guid.c_str(),
                           node_name.c_str(),
                           nullptr,
                           nullptr);
    assert(status == synSuccess && "Failed to call synNodeCreate()");

    //************************************************ compile graph
    synRecipeHandle recipeHandle = nullptr;
    std::string name("gather_2.recipe");
    status = synGraphCompile(&recipeHandle,
                             graphHandle,
                             name.c_str(),
                             nullptr);
    assert(status == synSuccess && "Failed to call synGraphCompile()");
#endif
    
    // ************************************************ runtime
    uint64_t workspace = 0;
    status = synWorkspaceGetSize(&workspace, recipeHandle);
    assert(status == synSuccess && "Failed to call synWorkspaceGetSize()");

    uint64_t hbm_addr = 0;
    status = synDeviceMalloc(deviceId, workspace, 0, 0, &hbm_addr);
    assert(status == synSuccess && "Failed to call synDeviceMalloc()");

    std::vector<void *> host_inputs;
    std::vector<void *> host_outputs;

    // malloc host X tensor
    uint64_t X_device = 0;
    status = synDeviceMalloc(deviceId, X_size, 0, 0, &X_device);
    assert(status == synSuccess && "Failed to call X synDeviceMalloc()");

    // malloc device X tensor
    unsigned char *X_host = nullptr;
    status = synHostMalloc(deviceId, X_size, 0, (void **)&X_host);
    assert(status == synSuccess && "Failed to call X synHostMalloc()");
    memset(X_host, 0, X_size);
    host_inputs.push_back(X_host);

    // malloc host I tensor
    uint64_t I_device = 0;
    status = synDeviceMalloc(deviceId, I_size, 0, 0, &I_device);
    assert(status == synSuccess && "Failed to call I synDeviceMalloc()");

    // malloc device I tensor
    unsigned char *I_host = nullptr;
    status = synHostMalloc(deviceId, I_size, 0, (void **)&I_host);
    assert(status == synSuccess && "Failed to call I synHostMalloc()");
    memset(I_host, 0, I_size);
    host_inputs.push_back(I_host);

    // malloc device Y tensor
    uint64_t Y_device = 0;
    status = synDeviceMalloc(deviceId, Y_size, 0, 0, &Y_device);
    assert(status == synSuccess && "Failed to call C synDeviceMalloc()");

    // malloc host Y tensor
    unsigned char *Y_host = nullptr;
    status = synHostMalloc(deviceId, Y_size, 0, (void **)&Y_host);
    assert(status == synSuccess && "Failed to call C synHostMalloc()");
    memset(Y_host, 0, Y_size);
    host_outputs.push_back(Y_host);

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

    status = synMemCopyAsync(h2d, (uint64_t)X_host, X_size, X_device, HOST_TO_DRAM);
    assert(status == synSuccess && "Failed to call X synMemCopyAsync()");

    status = synMemCopyAsync(h2d, (uint64_t)I_host, I_size, I_device, HOST_TO_DRAM);
    assert(status == synSuccess && "Failed to call I synMemCopyAsync()");
    
    status = synEventRecord(h2d_event, h2d);
    assert(status == synSuccess && "Failed to call synEventRecord()");

    status = synEventQuery(h2d_event);
    std::cout << "h2d_event status: " << status << std::endl;

    status = synStreamWaitEvent(compute, h2d_event, 0);
    assert(status == synSuccess && "Failed to call synEventRecord()");

    synLaunchTensorInfoExt X_launch;
    X_launch.tensorName = "X";
    X_launch.pTensorAddress = X_device;

    synLaunchTensorInfoExt I_launch;
    I_launch.tensorName = "I";
    I_launch.pTensorAddress = I_device;
    
    synLaunchTensorInfoExt Y_launch;
    Y_launch.tensorName = "Y";
    Y_launch.pTensorAddress = Y_device;

    std::vector<synLaunchTensorInfoExt> launch_info = {X_launch, I_launch, Y_launch};

    const char* tensorNames[3] = {};
    uint64_t    tensorIds[3];
    tensorNames[0] = launch_info[0].tensorName;
    tensorNames[1] = launch_info[1].tensorName;
    tensorNames[2] = launch_info[2].tensorName;
    synTensorRetrieveIds(recipeHandle, tensorNames, tensorIds, 3);
    launch_info[0].tensorId = tensorIds[0];
    launch_info[1].tensorId = tensorIds[1];
    launch_info[2].tensorId = tensorIds[2];
    
    status = synLaunchExt(compute, launch_info.data(), launch_info.size(), hbm_addr, recipeHandle, 0);

    status = synStreamSynchronize(compute);
    assert(status == synSuccess && "Failed to call synStreamSynchronize()");

    // copy data from device to host
    status = synMemCopyAsync(d2h, Y_device, Y_size, (uint64_t)Y_host, DRAM_TO_HOST);
    assert(status == synSuccess && "Failed to call C synMemCopyAsync()");

    status = synStreamSynchronize(d2h);
    assert(status == synSuccess && "Failed to call synStreamSynchronize()");

    // query device status
    status = synEventQuery(h2d_event);
    std::cout << "h2d_event status: " << status << std::endl;

    status = synStreamQuery(compute);
    std::cout << "compute stream status: " << status << std::endl;

    // clean
    status = synEventDestroy(h2d_event);
    assert(status == synSuccess && "Failed to call synEventDestroy()");

    status = synGraphDestroy(graphHandle);
    assert(status == synSuccess && "Failed to call synGraphDestroy()");

    status = synRecipeDestroy(recipeHandle);
    assert(status == synSuccess && "Failed to call synRecipeDestroy()");

    status = synHostFree(deviceId, X_host, 0);
    assert(status == synSuccess && "Failed to call synHostFree()");

    status = synHostFree(deviceId, I_host, 0);
    assert(status == synSuccess && "Failed to call synHostFree()");

    status = synHostFree(deviceId, Y_host, 0);
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
