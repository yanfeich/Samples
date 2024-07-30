make:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o unary unary.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	cp -rf unary ../../bin
dev:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o unary unary.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	cp -rf unary ../../bin
clean:
	rm -rf ../../bin/unary
	rm -rf unary
