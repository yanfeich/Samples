make:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o sdpa_recompute sdpa_recompute.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv sdpa_recompute ../../bin
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o sdpa sdpa.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv sdpa ../../bin
dev:
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o sdpa_recompute sdpa_recompute.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv sdpa_recompute ../../bin
	g++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o sdpa sdpa.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl
	mkdir -p ../../bin
	mv sdpa ../../bin
clean:
	rm -rf ../../bin/sdpa_recompute
	rm -rf ../../bin/sdpa
