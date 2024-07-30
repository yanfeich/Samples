make:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o gather gather.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv gather ../../bin
dev:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o gather gather.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv gather ../../bin
clean:
	rm -rf ../../bin/gather
