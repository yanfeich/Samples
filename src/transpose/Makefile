make:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o transpose transpose.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	cp -rf transpose ../../bin
dev:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o transpose transpose.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	cp -rf transpose ../../bin
clean:
	rm -rf ../../bin/transpose
	rm -rf transpose
