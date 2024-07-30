make:
	g++ -std=gnu++11 -I/usr/include/habanalabs \
        -I${SPDLOG_ROOT} -Wall -g -o sdpa sdpa.cpp \
        -L/usr/lib/habanalabs/ -lSynapse -ldl
	g++ -std=gnu++11 -I/usr/include/habanalabs \
        -I${SPDLOG_ROOT} -Wall -g -o sdpa_recompute sdpa_recompute.cpp \
        -L/usr/lib/habanalabs/ -lSynapse -ldl
dev:
	g++ -std=gnu++11 -I${SYNAPSE_ROOT}/include \
        -I${SYNAPSE_ROOT}/src/runtime/ \
        -I${SYNAPSE_ROOT}/src/common/ \
        -I${SYNAPSE_ROOT}/src/graph_compiler/ \
        -I${SPECS_EXT_ROOT}/ \
        -g -o sdpa sdpa.cpp \
        -L${SYNAPSE_RELEASE_BUILD}/lib/ -lSynapse -ldl
	g++ -std=gnu++11 -I${SYNAPSE_ROOT}/include \
        -I${SYNAPSE_ROOT}/src/runtime/ \
        -I${SYNAPSE_ROOT}/src/common/ \
        -I${SYNAPSE_ROOT}/src/graph_compiler/ \
        -I${SPECS_EXT_ROOT}/ \
        -g -o sdpa_recompute sdpa_recompute.cpp \
        -L${SYNAPSE_RELEASE_BUILD}/lib/ -lSynapse -ldl
clean:
	rm -f sdpa __repro
