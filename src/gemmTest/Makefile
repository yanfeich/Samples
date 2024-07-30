make:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o gemm gemm.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	cp -rf gemm ../../bin
dev:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o gemm gemm.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	cp -rf gemm ../../bin
clean:
	rm -rf ../../bin/gemm
	rm -rf gemm
