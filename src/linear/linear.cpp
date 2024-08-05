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
    unsigned B = 32;
    unsigned T = 20;
    unsigned IF = 11008;
    unsigned OF = 4096;

    std::map<synDataType, unsigned> dtype2size{
        {syn_type_fp8_143, 1},
        {syn_type_fp8_152, 1},
        {syn_type_bf16, 2},
        {syn_type_fp16, 2},
    };

    synStatus status = synInitialize();
    assert(status == synSuccess && "Failed to call synInitialize()");

    uint32_t deviceId = 0;
    synModuleId device_module_id = 1;
    status = synDeviceAcquireByModuleId(&deviceId, device_module_id);
    assert(status == synSuccess && "Failed to call synDeviceAcquireByModuleId()");

    synDeviceInfo deviceInfo;
    status = synDeviceGetInfo(deviceId, &deviceInfo);
    assert(status == synSuccess && "Failed to call synDeviceGetInfo()");

    synDeviceType device_type = deviceInfo.deviceType;

    std::vector<std::string> dtypes = {"bf16", "f16"};

    // bias=True
    for (size_t j = 0; j < dtypes.size(); j++)
    {
        synGraphHandle graphHandle = nullptr;
        status = synGraphCreate(&graphHandle, device_type);
        assert(status == synSuccess && "Failed to call synGraphCreate()");

        std::vector<synTensor> inputs;
        std::vector<synTensor> outputs;

        // ************************************************ create tensors
        unsigned int offset = 0;

        unsigned S_shape[] = {IF, T, B};
        unsigned S_size = B * T * IF;

        unsigned W_shape[] = {IF, OF};
        unsigned W_size = OF * IF;

        unsigned B_shape[] = {OF};
        unsigned B_size = OF;

        unsigned O_shape[] = {OF, T, B};
        unsigned O_size = B * T * OF;
        
        synDataType tensor_type = syn_type_fp16;

        if (dtypes[j] == "bf16")
        {
            S_size = S_size * 2;
            W_size = W_size * 2;
            B_size = B_size * 2;
            O_size = O_size * 2;
            tensor_type = syn_type_bf16;
        }
        else if (dtypes[j] == "f16")
        {
            S_size = S_size * 2;
            W_size = W_size * 2;
            B_size = B_size * 2;
            O_size = O_size * 2;
            tensor_type = syn_type_fp16;
        }

        // ************************************************ create input tensor
        synSectionHandle S_SectionHandle = nullptr;
        status = synSectionCreate(&S_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor S_desc;
        S_desc.m_dataType = tensor_type;
        S_desc.m_dims = 3;
        S_desc.m_name = "S";
        memset(S_desc.m_strides, 0, sizeof(S_desc.m_strides));
        memset(S_desc.m_sizes, 0, sizeof(S_desc.m_sizes));
        memcpy(S_desc.m_sizes, S_shape, 3 * sizeof(unsigned));

        synTensor syn_S = nullptr;
        status = synTensorCreate(&syn_S, &S_desc, S_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        inputs.push_back(syn_S);

        //************************************************ create weights tensor
        // NCHW => WHCN

        synSectionHandle W_SectionHandle = nullptr;
        status = synSectionCreate(&W_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor W_desc;
        W_desc.m_dataType = tensor_type;
        W_desc.m_dims = 2;
        W_desc.m_name = "W";
        memset(W_desc.m_strides, 0, sizeof(W_desc.m_strides));
        memset(W_desc.m_sizes, 0, sizeof(W_desc.m_sizes));
        memcpy(W_desc.m_sizes, W_shape, 2 * sizeof(unsigned));
        synTensor syn_W = nullptr;

        status = synTensorCreate(&syn_W, &W_desc, W_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        inputs.push_back(syn_W);

        //************************************************ create bias tensor

        synSectionHandle B_SectionHandle = nullptr;
        status = synSectionCreate(&B_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor B_desc;
        B_desc.m_dataType = tensor_type;
        B_desc.m_dims = 1;
        B_desc.m_name = "B";
        memset(B_desc.m_strides, 0, sizeof(B_desc.m_strides));
        memset(B_desc.m_sizes, 0, sizeof(B_desc.m_sizes));
        memcpy(B_desc.m_sizes, B_shape, 1 * sizeof(unsigned));
        synTensor syn_B = nullptr;
        status = synTensorCreate(&syn_B, &B_desc, B_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        inputs.push_back(syn_B);

        //************************************************ create output tensor
        // NCHW => WHCN

        synSectionHandle O_SectionHandle = nullptr;
        status = synSectionCreate(&O_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor O_desc;
        O_desc.m_dataType = tensor_type;
        O_desc.m_dims = 3;
        O_desc.m_name = "O";
        memset(O_desc.m_strides, 0, sizeof(O_desc.m_strides));
        memset(O_desc.m_sizes, 0, sizeof(O_desc.m_sizes));
        memcpy(O_desc.m_sizes, O_shape, 3 * sizeof(unsigned));
        synTensor syn_O = nullptr;

        status = synTensorCreate(&syn_O, &O_desc, O_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        outputs.push_back(syn_O);
        
        std::string guid = "linear_fwd_" + dtypes[j];
        std::string node_name = "linear";

        status = synNodeCreate(graphHandle,
                               inputs.data(),
                               outputs.data(),
                               3,
                               1,
                               nullptr,
                               0,
                               guid.c_str(),
                               node_name.c_str(),
                               nullptr,
                               nullptr);
        assert(status == synSuccess && "Failed to call synNodeCreate()");

        //************************************************ compile graph
        synRecipeHandle recipeHandle = nullptr;
        std::string name = "linear_bias_" + dtypes[j];
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

    // bias=False
    for (size_t j = 0; j < dtypes.size(); j++)
    {
        synGraphHandle graphHandle = nullptr;
        status = synGraphCreate(&graphHandle, device_type);
        assert(status == synSuccess && "Failed to call synGraphCreate()");

        std::vector<synTensor> inputs;
        std::vector<synTensor> outputs;

        // ************************************************ create tensors
        unsigned int offset = 0;

        unsigned S_shape[] = {IF, T, B};
        unsigned S_size = B * T * IF;

        unsigned W_shape[] = {IF, OF};
        unsigned W_size = OF * IF;

        unsigned O_shape[] = {OF, T, B};
        unsigned O_size = B * T * OF;
        
        synDataType tensor_type = syn_type_fp16;

        if (dtypes[j] == "bf16")
        {
            S_size = S_size * 2;
            W_size = W_size * 2;
            O_size = O_size * 2;
            tensor_type = syn_type_bf16;
        }
        else if (dtypes[j] == "f16")
        {
            S_size = S_size * 2;
            W_size = W_size * 2;
            O_size = O_size * 2;
            tensor_type = syn_type_fp16;
        }

        // ************************************************ create input tensor
        synSectionHandle S_SectionHandle = nullptr;
        status = synSectionCreate(&S_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor S_desc;
        S_desc.m_dataType = tensor_type;
        S_desc.m_dims = 3;
        S_desc.m_name = "S";
        memset(S_desc.m_strides, 0, sizeof(S_desc.m_strides));
        memset(S_desc.m_sizes, 0, sizeof(S_desc.m_sizes));
        memcpy(S_desc.m_sizes, S_shape, 3 * sizeof(unsigned));

        synTensor syn_S = nullptr;
        status = synTensorCreate(&syn_S, &S_desc, S_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        inputs.push_back(syn_S);

        //************************************************ create weights tensor
        // NCHW => WHCN

        synSectionHandle W_SectionHandle = nullptr;
        status = synSectionCreate(&W_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor W_desc;
        W_desc.m_dataType = tensor_type;
        W_desc.m_dims = 2;
        W_desc.m_name = "W";
        memset(W_desc.m_strides, 0, sizeof(W_desc.m_strides));
        memset(W_desc.m_sizes, 0, sizeof(W_desc.m_sizes));
        memcpy(W_desc.m_sizes, W_shape, 2 * sizeof(unsigned));
        synTensor syn_W = nullptr;

        status = synTensorCreate(&syn_W, &W_desc, W_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        inputs.push_back(syn_W);

        //************************************************ create output tensor
        // NCHW => WHCN

        synSectionHandle O_SectionHandle = nullptr;
        status = synSectionCreate(&O_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor O_desc;
        O_desc.m_dataType = tensor_type;
        O_desc.m_dims = 3;
        O_desc.m_name = "O";
        memset(O_desc.m_strides, 0, sizeof(O_desc.m_strides));
        memset(O_desc.m_sizes, 0, sizeof(O_desc.m_sizes));
        memcpy(O_desc.m_sizes, O_shape, 3 * sizeof(unsigned));
        synTensor syn_O = nullptr;

        status = synTensorCreate(&syn_O, &O_desc, O_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        outputs.push_back(syn_O);
        
        std::string guid = "linear_fwd_" + dtypes[j];
        std::string node_name = "linear";

        status = synNodeCreate(graphHandle,
                               inputs.data(),
                               outputs.data(),
                               2,
                               1,
                               nullptr,
                               0,
                               guid.c_str(),
                               node_name.c_str(),
                               nullptr,
                               nullptr);
        assert(status == synSuccess && "Failed to call synNodeCreate()");

        //************************************************ compile graph
        synRecipeHandle recipeHandle = nullptr;
        std::string name = "linear_" + dtypes[j];
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
    
    status = synDeviceRelease(deviceId);
    assert(status == synSuccess && "Failed to call synDeviceRelease()");

    status = synDestroy();
    assert(status == synSuccess && "Failed to call synDestroy()");
    return 0;
}
