CC = gcc
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Weffc++

main: main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lstdc++

clean:
	rm -f main

