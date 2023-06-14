#pragma once

std::string Exec(const char* cmd);

unsigned char RandomChar();

std::string GenerateHex(const unsigned int len);

void GetUniquesAndSort(std::vector<Node*> &v);

std::vector<std::vector<bool>> generateCombinations(int arity);
