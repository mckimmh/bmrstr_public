CC = clang++
CFLAGS = -Wall -O2 -I../include -std=c++17
LFLAGS = -larmadillo -lm -O2

################################################################################

bvg.out : bmrstr.o bvg.o log_post.o mvg.o regen_dist.o
	$(CC) $(LFLAGS) -o $@ $^

################################################################################

bmrstr.o : ../include/bmrstr.h ../include/log_post.h ../include/regen_dist.h \
           ../src/bmrstr.cpp
	$(CC) $(CFLAGS) -c ../src/bmrstr.cpp

bvg.o : bvg.cpp ../include/bmrstr.h ../include/log_post.h ../include/mvg.h \
        ../include/regen_dist.h
	$(CC) $(CFLAGS) -c bvg.cpp

log_post.o : ../include/log_post.h ../src/log_post.cpp
	$(CC) $(CFLAGS) -c ../src/log_post.cpp

mvg.o : ../include/mvg.h ../src/mvg.cpp
	$(CC) $(CFLAGS) -c ../src/mvg.cpp

regen_dist.o : ../include/regen_dist.h ../src/regen_dist.cpp
	$(CC) $(CFLAGS) -c ../src/regen_dist.cpp

.PHONY : clean
clean :
	rm *.out *.o *.txt