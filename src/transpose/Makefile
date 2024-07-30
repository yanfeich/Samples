make:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o transpose transpose.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv transpose ../../bin
dev:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o transpose transpose.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv transpose ../../bin
clean:
	rm -rf ../../bin/transpose
