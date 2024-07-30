make:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o cast cast.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv cast ../../bin
dev:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o cast cast.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv cast ../../bin
clean:
	rm -rf ../../bin/cast
