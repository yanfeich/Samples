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

    std::vector<std::string> dtypes = {"bf16", "f16", "f32"};

    for (size_t i = 0; i < dtypes.size(); i++)
    {
        synGraphHandle graphHandle = nullptr;
        status = synGraphCreate(&graphHandle, device_type);
        assert(status == synSuccess && "Failed to call synGraphCreate()");

        std::vector<synTensor> inputs;
        std::vector<synTensor> outputs;

        // ************************************************ create X tensor
        unsigned int offset = 0;
        unsigned X_shape[] = {H, T, B};
        unsigned X_size = H * T * B;

        unsigned Y_shape[] = {H, T, B};
        unsigned Y_size = H * T * B;

        synSectionHandle X_SectionHandle = nullptr;
        status = synSectionCreate(&X_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synDataType tensor_type = syn_type_single;
        if (dtypes[i] == "f32")
        {
            X_size = X_size * 4;
            Y_size = Y_size * 4;
            tensor_type = syn_type_single;
        }
        else if (dtypes[i] == "f16")
        {
            X_size = X_size * 2;
            Y_size = Y_size * 2;
            tensor_type = syn_type_bf16;
        }
        else if (dtypes[i] == "bf16")
        {
            X_size = X_size * 2;
            Y_size = Y_size * 2;
            tensor_type = syn_type_fp16;
        }

        synTensorDescriptor X_desc;
        X_desc.m_dataType = tensor_type;
        X_desc.m_dims = 3UL;
        X_desc.m_name = "X";
        memset(X_desc.m_strides, 0, sizeof(X_desc.m_strides));
        memset(X_desc.m_sizes, 0, sizeof(X_desc.m_sizes));
        memcpy(X_desc.m_sizes, X_shape, 3 * sizeof(unsigned));

        synTensor syn_X = nullptr;
        status = synTensorCreate(&syn_X, &X_desc, X_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        inputs.push_back(syn_X);

        //************************************************ create Gamma tensor
        unsigned gamma_shape[] = {H};
        unsigned gamma_size = H * 4;
        synSectionHandle gamma_SectionHandle = nullptr;
        status = synSectionCreate(&gamma_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor gamma_desc;
        gamma_desc.m_dataType = syn_type_single;
        gamma_desc.m_dims = 1UL;
        gamma_desc.m_name = "Gamma";
        memset(gamma_desc.m_strides, 0, sizeof(gamma_desc.m_strides));
        memset(gamma_desc.m_sizes, 0, sizeof(gamma_desc.m_sizes));
        memcpy(gamma_desc.m_sizes, gamma_shape, 1 * sizeof(unsigned));
        synTensor syn_gamma = nullptr;
        status = synTensorCreate(&syn_gamma, &gamma_desc, gamma_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        inputs.push_back(syn_gamma);

        //************************************************ create C tensor
        synSectionHandle Y_SectionHandle = nullptr;
        status = synSectionCreate(&Y_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor Y_desc;
        Y_desc.m_dataType = tensor_type;
        Y_desc.m_dims = 3UL;
        Y_desc.m_name = "Y";
        memset(Y_desc.m_strides, 0, sizeof(Y_desc.m_strides));
        memset(Y_desc.m_sizes, 0, sizeof(Y_desc.m_sizes));
        memcpy(Y_desc.m_sizes, Y_shape, 3 * sizeof(unsigned));
        synTensor syn_Y = nullptr;

        status = synTensorCreate(&syn_Y, &Y_desc, Y_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        outputs.push_back(syn_Y);

        //************************************************ create gamma grad tensor
        unsigned mean_square_shape[] = {1, T, B};
        unsigned mean_square_size = B * T * 4;
        synSectionHandle mean_square_SectionHandle = nullptr;
        status = synSectionCreate(&mean_square_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor mean_square;
        mean_square.m_dataType = syn_type_single;
        mean_square.m_dims = 3UL;
        mean_square.m_name = "mean_square";
        memset(mean_square.m_strides, 0, sizeof(Y_desc.m_strides));
        memset(mean_square.m_sizes, 0, sizeof(Y_desc.m_sizes));
        memcpy(mean_square.m_sizes, mean_square_shape, 3 * sizeof(unsigned));
        synTensor syn_mean_square = nullptr;

        status = synTensorCreate(&syn_mean_square, &mean_square, mean_square_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        outputs.push_back(syn_mean_square);

        //************************************************ create rms node
        std::string guid = "rms_norm_ex_fwd_" + dtypes[i];
        std::string node_name = "RMS";
        ns_LayerNormKernel::Params rms_params;
        rms_params.epsValid = true;
        rms_params.eps = float(1e-5);

        status = synNodeCreate(graphHandle,
                               inputs.data(),
                               outputs.data(),
                               2,
                               2,
                               &rms_params,
                               sizeof(rms_params),
                               guid.c_str(),
                               node_name.c_str(),
                               nullptr,
                               nullptr);
        assert(status == synSuccess && "Failed to call synNodeCreate()");

        //************************************************ compile graph
        synRecipeHandle recipeHandle = nullptr;
        std::string name = guid + "recipe";
        status = synGraphCompile(&recipeHandle,
                                 graphHandle,
                                 name.c_str(),
                                 nullptr);
        assert(status == synSuccess && "Failed to call synGraphCompile()");

        status = synGraphDestroy(graphHandle);
        assert(status == synSuccess && "Failed to call synGraphDestroy()");

        status = synRecipeDestroy(recipeHandle);
        assert(status == synSuccess && "Failed to call synRecipeDestroy()");
    }

    // clean

    status = synDeviceRelease(deviceId);
    assert(status == synSuccess && "Failed to call synDeviceRelease()");

    status = synDestroy();
    assert(status == synSuccess && "Failed to call synDestroy()");
    return 0;
}
