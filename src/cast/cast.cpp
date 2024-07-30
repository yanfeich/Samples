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

int main(int argc, char *argv[])
{
    unsigned B = 64;
    unsigned T = 128;
    unsigned H = 4096;

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
        {"hf8",  syn_type_fp8_143},
        {"f8",   syn_type_fp8_152},
        {"i16",  syn_type_int16},
        {"u16",  syn_type_uint16},
        {"bf16", syn_type_bf16},
        {"f16",  syn_type_fp16},
        {"f32",  syn_type_single},
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
    std::map<std::string, synDataType>::iterator it_from;
    std::map<std::string, synDataType>::iterator it_to;
    std::map<synDataType, unsigned>::iterator iter;
    for (it_from = type2dtype.begin(); it_from != type2dtype.end(); it_from++)
    {
        for (it_to = type2dtype.begin(); it_to != type2dtype.end(); it_to++)
        {
            if((*it_from).first == (*it_to).first)
                continue;
            
            if(((*it_from).first =="f8") && ((((*it_to).first) == "hf8")
                || (((*it_to).first) == "i8")  || (((*it_to).first) == "u8")
                || (((*it_to).first) == "i16") || (((*it_to).first) == "u16")
                || (((*it_to).first) == "f16")))
                continue;
            if(((*it_from).first =="hf8") && ((((*it_to).first) == "f8")
                || (((*it_to).first) == "i8")  || (((*it_to).first) == "u8")
                || (((*it_to).first) == "i16") || (((*it_to).first) == "u16")
                || (((*it_to).first) == "f16")))
                continue;

            if((((*it_to).first) == "f8") && 
                  (((*it_from).first =="i8")  || ((*it_from).first =="u8")
                || ((*it_from).first =="i16") || ((*it_from).first =="u16")
                || ((*it_from).first =="f16")))
                continue;
            if((((*it_to).first) == "hf8") && 
                  (((*it_from).first =="i8")  || ((*it_from).first =="u8")
                || ((*it_from).first =="i16") || ((*it_from).first =="u16")
                || ((*it_from).first =="f16")))
                continue;

            synGraphHandle graphHandle = nullptr;
            status = synGraphCreate(&graphHandle, device_type);
            assert(status == synSuccess && "Failed to call synGraphCreate()");
           
            std::vector<synTensor> inputs;
            std::vector<synTensor> outputs;
    
            // ************************************************ create X tensor
            unsigned int offset = 0;
            unsigned X_shape[] = {H, T, B};
            iter = dtype2size.find((*it_from).second);
            unsigned X_size = H * T * B * (*iter).second;
            synSectionHandle X_SectionHandle = nullptr;
            status = synSectionCreate(&X_SectionHandle, 0, graphHandle);
            assert(status == synSuccess && "Failed to call synSectionCreate()");
            
            synTensorDescriptor X_desc;
            X_desc.m_dataType = (*it_from).second;
            X_desc.m_dims = 3UL;
            X_desc.m_name = "X";
            memset(X_desc.m_strides, 0, sizeof(X_desc.m_strides));
            memset(X_desc.m_sizes, 0, sizeof(X_desc.m_sizes));
            memcpy(X_desc.m_sizes, X_shape, 3 * sizeof(unsigned));
            
            synTensor syn_X = nullptr;
            status = synTensorCreate(&syn_X, &X_desc, X_SectionHandle, offset);
            assert(status == synSuccess && "Failed to call synTensorCreate()");
            inputs.push_back(syn_X);
            
            //************************************************ create Y tensor
            unsigned Y_shape[] = {H, T, B};
            iter = dtype2size.find((*it_to).second);
            unsigned Y_size = H * T * B * (*iter).second;
            synSectionHandle Y_SectionHandle = nullptr;
            status = synSectionCreate(&Y_SectionHandle, 0, graphHandle);
            assert(status == synSuccess && "Failed to call synSectionCreate()");
            
            synTensorDescriptor Y_desc;
            Y_desc.m_dataType = (*it_to).second;
            Y_desc.m_dims = 3UL;
            Y_desc.m_name = "Y";
            memset(Y_desc.m_strides, 0, sizeof(Y_desc.m_strides));
            memset(Y_desc.m_sizes, 0, sizeof(Y_desc.m_sizes));
            memcpy(Y_desc.m_sizes, Y_shape, 3 * sizeof(unsigned));
            synTensor syn_Y = nullptr;
            
            status = synTensorCreate(&syn_Y, &Y_desc, Y_SectionHandle, offset);
            assert(status == synSuccess && "Failed to call synTensorCreate()");
            outputs.push_back(syn_Y);

            //************************************************ create cast node
            std::string guid = "cast_" + (*it_from).first + "_to_" + (*it_to).first;
            std::string node_name = "CAST_1";
            
            ns_CastKernel::Params cast_params;
            cast_params.round_mode = CAST_ROUND_HALF_NE;
            
            status = synNodeCreate(graphHandle,
                                   inputs.data(),
                                   outputs.data(),
                                   1,
                                   1,
                                   &cast_params,
                                   sizeof(cast_params),
                                   guid.c_str(),
                                   node_name.c_str(),
                                   nullptr,
                                   nullptr);
            assert(status == synSuccess && "Failed to call synNodeCreate()");
            
            //************************************************ compile graph
            synRecipeHandle recipeHandle = nullptr;
            std::string name(guid + ".recipe");
            status = synGraphCompile(&recipeHandle,
                                     graphHandle,
                                     name.c_str(),
                                     nullptr);
            assert(status == synSuccess && "Failed to call synGraphCompile()");
            
            // clean
            status = synGraphDestroy(graphHandle);
            assert(status == synSuccess && "Failed to call synGraphDestroy()");
          
            status = synRecipeDestroy(recipeHandle);
            assert(status == synSuccess && "Failed to call synRecipeDestroy()");

        }
    }

    status = synDeviceRelease(deviceId);
    assert(status == synSuccess && "Failed to call synDeviceRelease()");

    status = synDestroy();
    assert(status == synSuccess && "Failed to call synDestroy()");
    return 0;
}
