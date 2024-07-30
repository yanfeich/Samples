make:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o binary binary.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv binary ../../bin
dev:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o binary binary.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv binary ../../bin
clean:
	rm -rf ../../bin/binary
