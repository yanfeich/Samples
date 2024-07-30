make:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o batch_gemm batch_gemm.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv batch_gemm ../../bin
dev:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o batch_gemm batch_gemm.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv batch_gemm ../../bin
clean:
	rm -rf ../../bin/batch_gemm
