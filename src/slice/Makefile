make:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o slice slice.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	cp -rf slice ../../bin
dev:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o slice slice.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	cp -rf slice ../../bin
clean:
	rm -rf ../../bin/slice
	rm -rf slice
