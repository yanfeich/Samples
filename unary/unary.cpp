#include <time.h>
#include <unistd.h>
#include <string.h>
#include <algorithm>
#include <synapse_api.h>
#include <synapse_common_types.hpp>
#include <cassert>
#include <iostream>
#include <vector>
#include <map>

int main(int argc, char *argv[])
{
    unsigned B = 64;
    unsigned T = 128;
    unsigned H = 4096;

    std::map<synDataType, unsigned> dtype2size{
        {syn_type_bf16, 2}};

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

    // ************************************************ create X tensor
    unsigned int offset = 0;
    unsigned X_shape[] = {H, T, B};
    unsigned X_size = H * T * B * 2;
    synSectionHandle X_SectionHandle = nullptr;
    status = synSectionCreate(&X_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor X_desc;
    X_desc.m_dataType = syn_type_bf16;
    X_desc.m_dims = 3UL;
    X_desc.m_name = "X";
    memset(X_desc.m_strides, 0, sizeof(X_desc.m_strides));
    memset(X_desc.m_sizes, 0, sizeof(X_desc.m_sizes));
    memcpy(X_desc.m_sizes, X_shape, 3 * sizeof(unsigned));

    synTensor syn_X = nullptr;
    status = synTensorCreate(&syn_X, &X_desc, X_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    inputs.push_back(syn_X);

    unsigned Y_shape[] = {H, T, B};
    unsigned Y_size = H * T * B * 2;

    //************************************************ create rms node
    std::vector<std::string> unary_guids = {"relu", "abs", "exp", "sigmoid", "sign", "sqrt",
                                            "tanh", "erf", "log", "neg", "tan", "softsign"};
    std::vector<std::string> dtypes = {"bf16", "f32", "f16"};

    for (size_t i = 0; i < unary_guids.size(); i++)
    {
        for (size_t j = 0; j < dtypes.size(); j++)
        {
            if (dtypes[j] == "f16" || dtypes[j] == "bf16")
            {
                if (unary_guids[i] == "erf" ||
                    unary_guids[i] == "tan" ||
                    unary_guids[i] == "tanh" ||
                    unary_guids[i] == "log")
                {
                    continue;
                }
            }

            std::vector<synTensor> outputs;
            std::string guid = unary_guids[i] + "_" + dtypes[j];

            synSectionHandle Y_SectionHandle = nullptr;
            status = synSectionCreate(&Y_SectionHandle, 0, graphHandle);
            assert(status == synSuccess && "Failed to call synSectionCreate()");

            synTensorDescriptor Y_desc;
            Y_desc.m_dataType = syn_type_bf16;
            Y_desc.m_dims = 3UL;
            Y_desc.m_name = std::string(guid + "Y").c_str();
            memset(Y_desc.m_strides, 0, sizeof(Y_desc.m_strides));
            memset(Y_desc.m_sizes, 0, sizeof(Y_desc.m_sizes));
            memcpy(Y_desc.m_sizes, Y_shape, 3 * sizeof(unsigned));
            synTensor syn_Y = nullptr;

            status = synTensorCreate(&syn_Y, &Y_desc, Y_SectionHandle, offset);
            assert(status == synSuccess && "Failed to call synTensorCreate()");
            outputs.push_back(syn_Y);

            std::string node_name = guid + "_" + std::to_string(i) + "_" + std::to_string(j);
            status = synNodeCreate(graphHandle,
                                   inputs.data(),
                                   outputs.data(),
                                   1,
                                   1,
                                   nullptr,
                                   0,
                                   guid.c_str(),
                                   node_name.c_str(),
                                   nullptr,
                                   nullptr);
            assert(status == synSuccess && "Failed to call synNodeCreate()");
        }
    }

    //************************************************ compile graph
    synRecipeHandle recipeHandle = nullptr;
    std::string name("unary.recipe");
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

    // malloc device C tensor
    uint64_t Y_device = 0;
    status = synDeviceMalloc(deviceId, Y_size, 0, 0, &Y_device);
    assert(status == synSuccess && "Failed to call C synDeviceMalloc()");

    // malloc host C tensor
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

    status = synEventRecord(h2d_event, h2d);
    assert(status == synSuccess && "Failed to call synEventRecord()");

    status = synEventQuery(h2d_event);
    std::cout << "h2d_event status: " << status << std::endl;

    status = synStreamWaitEvent(compute, h2d_event, 0);
    assert(status == synSuccess && "Failed to call synEventRecord()");

    synLaunchTensorInfoExt A_launch;
    A_launch.tensorName = "X";
    A_launch.pTensorAddress = X_device;

    synLaunchTensorInfoExt C_launch;
    C_launch.tensorName = "Y";
    C_launch.pTensorAddress = Y_device;

    std::vector<synLaunchTensorInfoExt> launch_info = {A_launch, C_launch};

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
