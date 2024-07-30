make:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o index_copy index_copy.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	cp -rf index_copy ../../bin
dev:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o index_copy index_copy.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	cp -rf index_copy ../../bin
clean:
	rm -rf ../../bin/index_copy
	rm -rf index_copy
