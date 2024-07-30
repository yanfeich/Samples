#include <time.h>
#include <unistd.h>
#include <string.h>
#include <algorithm>
#include <synapse_api.h>
#include <perf_lib_layer_params.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <map>

int main(int argc, char *argv[])
{
    unsigned L = 256;

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
    std::vector<synTensor> mid;
    std::vector<synTensor> outputs;

    // ************************************************ create X tensor
    unsigned int offset = 0;
    unsigned X_shape[] = {L};
    unsigned X_size = L * 4;
    synSectionHandle X_SectionHandle = nullptr;
    status = synSectionCreate(&X_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor X_desc;
    X_desc.m_dataType = syn_type_single;
    X_desc.m_dims = 1UL;
    X_desc.m_name = "X";
    memset(X_desc.m_strides, 0, sizeof(X_desc.m_strides));
    memset(X_desc.m_sizes, 0, sizeof(X_desc.m_sizes));
    memcpy(X_desc.m_sizes, X_shape, 4 * sizeof(unsigned));

    synTensor syn_X = nullptr;
    status = synTensorCreate(&syn_X, &X_desc, X_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    inputs.push_back(syn_X);

    //************************************************ create B tensor
    unsigned B_shape[] = {L};
    unsigned B_size = L * 4;
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor B_desc;
    B_desc.m_dataType = syn_type_single;
    B_desc.m_dims = 1UL;
    B_desc.m_name = "B";
    memset(B_desc.m_strides, 0, sizeof(B_desc.m_strides));
    memset(B_desc.m_sizes, 0, sizeof(B_desc.m_sizes));
    memcpy(B_desc.m_sizes, B_shape, 2 * sizeof(unsigned));
    synTensor syn_B = nullptr;
    status = synTensorCreate(&syn_B, &B_desc, nullptr, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    mid.push_back(syn_B);

    //************************************************ create Y tensor
    // NCHW => WHCN
    unsigned Y_shape[] = {L};
    unsigned Y_size = L * 4;
    synSectionHandle Y_SectionHandle = nullptr;
    status = synSectionCreate(&Y_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor Y_desc;
    Y_desc.m_dataType = syn_type_single;
    Y_desc.m_dims = 1UL;
    Y_desc.m_name = "Y";
    memset(Y_desc.m_strides, 0, sizeof(Y_desc.m_strides));
    memset(Y_desc.m_sizes, 0, sizeof(Y_desc.m_sizes));
    memcpy(Y_desc.m_sizes, Y_shape, 4 * sizeof(unsigned));
    synTensor syn_Y = nullptr;

    status = synTensorCreate(&syn_Y, &Y_desc, Y_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    outputs.push_back(syn_Y);

    //************************************************ create relu node
    std::string guid = "relu_f32";
    std::string node_name = "RELU_1";

    ns_ReluKernel::ParamsV2 relu_params;
    relu_params.threshold.f = 0.0;
    relu_params.replacementValue.f = -0.1;

    status = synNodeCreate(graphHandle,
                           inputs.data(),
                           mid.data(),
                           1,
                           1,
                           &relu_params,
                           sizeof(relu_params),
                           guid.c_str(),
                           node_name.c_str(),
                           nullptr,
                           nullptr);
    std::cout << "ReLU synNodeCreate status: " << status << std::endl;
    assert(status == synSuccess && "Failed to call synNodeCreate()");

    node_name = "RELU_2";
    status = synNodeCreate(graphHandle,
                           mid.data(),
                           outputs.data(),
                           1,
                           1,
                           &relu_params,
                           sizeof(relu_params),
                           guid.c_str(),
                           node_name.c_str(),
                           nullptr,
                           nullptr);
    std::cout << "ReLU synNodeCreate status: " << status << std::endl;
    assert(status == synSuccess && "Failed to call synNodeCreate()");
    //************************************************ compile graph
    synRecipeHandle recipeHandle = nullptr;
    std::string name("relu.recipe");
    status = synGraphCompile(&recipeHandle,
                             graphHandle,
                             name.c_str(),
                             nullptr);
    assert(status == synSuccess && "Failed to call synGraphCompile()");

    // ************************************************ runtime
    uint64_t workspace = 0;
    status = synWorkspaceGetSize(&workspace, recipeHandle);
    std::cout << "workspace size = " << workspace << std::endl;
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
    //memset(X_host, 0, X_size);
    float *float_ptr = (float*)X_host;
    for (int j = 0; j < X_size/4; j++)
    {
        float_ptr[j] = (std::rand() - RAND_MAX/2) / float(RAND_MAX);
    }
    host_inputs.push_back(X_host);

    // malloc device Y tensor
    uint64_t Y_device = 0;
    status = synDeviceMalloc(deviceId, Y_size, 0, 0, &Y_device);
    assert(status == synSuccess && "Failed to call Y synDeviceMalloc()");

    // malloc host Y tensor
    unsigned char *Y_host = nullptr;
    status = synHostMalloc(deviceId, Y_size, 0, (void **)&Y_host);
    assert(status == synSuccess && "Failed to call Y synHostMalloc()");
    memset(Y_host, 255, Y_size);
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

    synEventHandle compute_event = nullptr;
    status = synEventCreate(&compute_event, deviceId, 0);
    assert(status == synSuccess && "Failed to call synEventCreate()");
    
    status = synMemCopyAsync(h2d, (uint64_t)X_host, X_size, X_device, HOST_TO_DRAM);
    assert(status == synSuccess && "Failed to call X synMemCopyAsync()");

    status = synMemCopyAsync(h2d, (uint64_t)Y_host, Y_size, Y_device, HOST_TO_DRAM);
    assert(status == synSuccess && "Failed to call Y synMemCopyAsync()");

    status = synEventRecord(h2d_event, h2d);
    assert(status == synSuccess && "Failed to call synEventRecord()");

    status = synEventQuery(h2d_event);
    std::cout << "h2d_event status: " << status << std::endl;
    
    status = synStreamWaitEvent(compute, h2d_event, 0);
    assert(status == synSuccess && "Failed to call synEventRecord()");
    // std::vector<synLaunchTensorInfoExt> *graph_inputs = nullptr;
    // std::vector<synLaunchTensorInfoExt> *graph_outputs = nullptr;
    synLaunchTensorInfoExt X_launch;
    X_launch.tensorName = "X";
    X_launch.pTensorAddress = X_device;
    //X_launch.tensorId = 0;

    synLaunchTensorInfoExt Y_launch;
    Y_launch.tensorName = "Y";
    Y_launch.pTensorAddress = Y_device;
    //Y_launch.tensorId = 1;
    
    std::vector<synLaunchTensorInfoExt> launch_info = {X_launch, /*B_launch,*/ Y_launch};

    // retrieve mapping of tensorName <--> tensorId
    const char* tensorNames[2] = {};
    uint64_t    tensorIds[2];
    tensorNames[0] = launch_info[0].tensorName;
    tensorNames[1] = launch_info[1].tensorName;
    synTensorRetrieveIds(recipeHandle, tensorNames, tensorIds, 2);
    launch_info[0].tensorId = tensorIds[0];
    launch_info[1].tensorId = tensorIds[1];
    std::cout << tensorIds[0] << std::endl;
    std::cout << tensorIds[1] << std::endl;
    
    status = synLaunchExt(compute, launch_info.data(), launch_info.size(), hbm_addr, recipeHandle, 0);
    assert(status == synSuccess && "Failed to call synLaunchExt()");

    status = synEventRecord(compute_event, compute);
    assert(status == synSuccess && "Failed to call synEventRecord()");
    
    status = synEventSynchronize(compute_event);
    status = synStreamWaitEvent(d2h, compute_event, 0);
    status = synStreamSynchronize(compute);
    assert(status == synSuccess && "Failed to call synStreamSynchronize()");

    status = synStreamQuery(compute);
    std::cout << "compute stream status: " << status << std::endl;

    // copy data from device to host
    status = synMemCopyAsync(d2h, Y_device, Y_size, (uint64_t)Y_host, DRAM_TO_HOST);
    std::cout << "copy Y from device to host status: " << status << std::endl;
    assert(status == synSuccess && "Failed to call Y synMemCopyAsync()");

    status = synStreamSynchronize(d2h);
    assert(status == synSuccess && "Failed to call synStreamSynchronize()");

    // query device status
    status = synEventQuery(h2d_event);
    std::cout << "h2d_event status: " << status << std::endl;

    float_ptr = (float*)X_host;
    for (int j = 0; j < X_size/4; j++)
    {
        std::cout << float_ptr[j] << "\t";
    }
    std::cout << std::endl;
    float_ptr = (float*)Y_host;
    for (int j = 0; j < Y_size/4; j++)
    {
        std::cout << float_ptr[j] << "\t";
    }
    std::cout << std::endl;
    
    // clean
    status = synEventDestroy(h2d_event);
    assert(status == synSuccess && "Failed to call synEventDestroy()");
    
    status = synEventDestroy(compute_event);
    assert(status == synSuccess && "Failed to call synEventDestroy()");

    status = synGraphDestroy(graphHandle);
    assert(status == synSuccess && "Failed to call synGraphDestroy()");

    status = synRecipeDestroy(recipeHandle);
    assert(status == synSuccess && "Failed to call synRecipeDestroy()");

    status = synHostFree(deviceId, X_host, 0);
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
