make:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o rope rope.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	cp -rf rope ../../bin
dev:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o rope rope.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	cp -rf rope ../../bin
clean:
	rm -rf ../../bin/rope
	rm -rf rope
