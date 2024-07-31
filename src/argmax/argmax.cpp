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
    unsigned B = 32;
    unsigned S = 32000;

    std::map<synDataType, unsigned> dtype2size{
        {syn_type_int8, 1},
        {syn_type_uint8, 1},
        {syn_type_fp8_143, 1},
        {syn_type_fp8_152, 1},
        {syn_type_int16, 2},
        {syn_type_uint16, 2},
        {syn_type_bf16, 2},
        {syn_type_fp16, 2},
        {syn_type_uint32, 4},
        {syn_type_int32, 4},
        {syn_type_uint32, 4},
        {syn_type_single, 4},
    };

    std::map<std::string, synDataType> type2dtype{
        {"i8",   syn_type_int8},
        {"u8",   syn_type_uint8},
        {"i32",  syn_type_int32},
        {"f32",  syn_type_single},
        {"bf16", syn_type_bf16},
        {"f16",  syn_type_fp16},
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

    std::map<std::string, synDataType>::iterator it_s;
    std::map<synDataType, unsigned>::iterator iter;
    for (it_s = type2dtype.begin(); it_s != type2dtype.end(); it_s++)
    {
        synGraphHandle graphHandle = nullptr;
        status = synGraphCreate(&graphHandle, device_type);
        assert(status == synSuccess && "Failed to call synGraphCreate()");

        std::vector<synTensor> inputs;
        std::vector<synTensor> outputs;

        // ************************************************ create tensors
        unsigned int offset = 0;
        unsigned int N_dim = 2;
        unsigned int dim = 0;

        unsigned S_shape[] = {S, B};
        iter = dtype2size.find((*it_s).second);
        unsigned S_size = B * S * (*iter).second;

        unsigned O_shape[] = {S, B};
        if(dim == -1)
            dim = 0;
        O_shape[dim] = 1;

        unsigned O_size = 4;
        for (int i = 0; i < N_dim; i++)
            O_size = O_size * O_shape[i];
        
        // ************************************************ create input tensor
        synSectionHandle S_SectionHandle = nullptr;
        status = synSectionCreate(&S_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor S_desc;
        S_desc.m_dataType = (*it_s).second;
        S_desc.m_dims = 2;
        S_desc.m_name = "S";
        memset(S_desc.m_strides, 0, sizeof(S_desc.m_strides));
        memset(S_desc.m_sizes, 0, sizeof(S_desc.m_sizes));
        memcpy(S_desc.m_sizes, S_shape, 2 * sizeof(unsigned));

        synTensor syn_S = nullptr;
        status = synTensorCreate(&syn_S, &S_desc, S_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        inputs.push_back(syn_S);

        //************************************************ create output tensor
        synSectionHandle O_SectionHandle = nullptr;
        status = synSectionCreate(&O_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor O_desc;
        O_desc.m_dataType = syn_type_int32;
        O_desc.m_dims = 2;
        O_desc.m_name = "O";
        memset(O_desc.m_strides, 0, sizeof(O_desc.m_strides));
        memset(O_desc.m_sizes, 0, sizeof(O_desc.m_sizes));
        memcpy(O_desc.m_sizes, O_shape, 2 * sizeof(unsigned));
        synTensor syn_O = nullptr;

        status = synTensorCreate(&syn_O, &O_desc, O_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        outputs.push_back(syn_O);
        
        std::string guid = "argmax_fwd_" + (*it_s).first;
        std::string node_name = "Argmax";

        ns_Reduction::Params reduce_params;
        reduce_params.reductionDimension = dim;
    
        status = synNodeCreate(graphHandle,
                               inputs.data(),
                               outputs.data(),
                               1,
                               1,
                               &reduce_params,
                               sizeof(reduce_params),
                               guid.c_str(),
                               node_name.c_str(),
                               nullptr,
                               nullptr);
        assert(status == synSuccess && "Failed to call synNodeCreate()");

        //************************************************ compile graph
        synRecipeHandle recipeHandle = nullptr;
        std::string name = guid;
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
