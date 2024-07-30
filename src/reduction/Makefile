make:
	g++ -std=gnu++11 -I/usr/include/habanalabs \
        -I${SPDLOG_ROOT} -Wall -g -o reduction reduction.cpp \
        -L/usr/lib/habanalabs/ -lSynapse -ldl

dev:
	g++ -std=gnu++11 -I${SYNAPSE_ROOT}/include \
        -I${SYNAPSE_ROOT}/src/runtime/ \
        -I${SYNAPSE_ROOT}/src/common/ \
        -I${SYNAPSE_ROOT}/src/graph_compiler/ \
        -I${SPECS_EXT_ROOT}/ \
        -g -o reduction reduction.cpp \
        -L${SYNAPSE_RELEASE_BUILD}/lib/ -lSynapse -ldl

clean:
	rm -f reduction __repro
