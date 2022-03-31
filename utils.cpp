#include "utils.hpp"

// Taken from https://stackoverflow.com/questions/478898/how-do-i-execute-a-command-and-get-the-output-of-the-command-within-c-using-po
std::string Exec(const char* cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

// Begin
// Taken from https://lowrey.me/guid-generation-in-c-11/
unsigned char RandomChar() {
  std::random_device rd;
  std::mt19937 gen(rd()); 
  std::uniform_int_distribution<> dis(0, 255);
  return static_cast<unsigned char>(dis(gen));
}

std::string GenerateHex(const unsigned int len) {
  std::stringstream ss;
  for(auto i = 0; i < len; i++) {
    auto rc = RandomChar();
    std::stringstream hexstream;
    hexstream << std::hex << int(rc);
    auto hex = hexstream.str(); 
    ss << (hex.length() < 2 ? '0' + hex : hex);
  }        
  return ss.str();
}
// End

// Taken from https://stackoverflow.com/questions/16476099/remove-duplicate-entries-in-a-c-vector
void GetUniquesAndSort(std::vector<Node*> &v) {
  std::sort(v.begin(), v.end()); 
  auto last = std::unique(v.begin(), v.end());
  v.erase(last, v.end());
}
