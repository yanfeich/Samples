make:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o relu relu.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv relu ../../bin
dev:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o relu relu.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv relu ../../bin
clean:
	rm -rf ../../bin/relu
