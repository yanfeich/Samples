make:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o reduction reduction.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv reduction ../../bin
dev:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o reduction reduction.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv reduction ../../bin
clean:
	rm -rf ../../bin/reduction
