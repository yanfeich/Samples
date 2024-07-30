make:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o cast cast.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	cp -rf cast ../../bin
dev:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o cast cast.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	cp -rf cast ../../bin
clean:
	rm -rf ../../bin/cast
	rm -rf cast
