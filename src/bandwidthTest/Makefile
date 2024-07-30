make:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o bandwidth bandwidth.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv bandwidth ../../bin
dev:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o bandwidth bandwidth.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv bandwidth ../../bin
clean:
	rm -rf ../../bin/bandwidth
