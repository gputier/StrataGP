// tools/qwen35/q36_tokfile.cpp - strata-q36 with `@FILE` arguments: an argument that starts with '@' is replaced by the
// content of that file (trailing whitespace dropped).  A prompt of tens of thousands of token ids does not fit the 32767
// characters of a Windows command line; `--tokens @ids.txt` does.  The engine is unchanged: this calls the same
// strata::qwen35::run as strata-q36.
//
// Built next to strata-q36 (it needs strata_kq.dll from the same folder), linked as strata-q36 is, with this object in
// place of src/qwen35/q36_main.cpp's:
//   cl /nologo /EHsc /O2 /std:c++20 /I include /c tools/qwen35/q36_tokfile.cpp
#include "strata/qwen35/engine.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (i > 0 && a.size() > 1 && a[0] == '@') {
            std::ifstream f(a.substr(1), std::ios::binary);
            if (!f) {
                std::fprintf(stderr, "q36_tokfile: cannot read %s\n", a.c_str() + 1);
                return 2;
            }
            a.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            while (!a.empty() && (a.back() == '\n' || a.back() == '\r' || a.back() == ' ')) a.pop_back();
        }
        args.push_back(std::move(a));
    }
    std::vector<char*> ptrs;
    for (std::string& s : args) ptrs.push_back(s.data());
    ptrs.push_back(nullptr);
    return strata::qwen35::run((int) args.size(), ptrs.data());
}
