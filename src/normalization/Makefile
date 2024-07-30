make:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o rms rms.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv rms ../../bin
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o layernorm layernorm.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv layernorm ../../bin
dev:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o rms rms.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv rms ../../bin
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o layernorm layernorm.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv layernorm ../../bin
clean:
	rm -rf ../../bin/rms
	rm -rf ../../bin/layernorm
