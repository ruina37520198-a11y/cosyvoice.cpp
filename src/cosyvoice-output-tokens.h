#pragma once
#include <vector>
#include <cstdint>
// Generated history belongs to the LLM; this buffer alone feeds the vocoder.
struct cosyvoice_output_tokens {
    std::vector<int> tokens;
    uint32_t silent_run = 0;
    void accept(int token, bool silent) {
        silent_run = silent ? silent_run + 1 : 0;
        if (silent_run <= 5) tokens.push_back(token);
    }
    void clear() { tokens.clear(); silent_run = 0; }
};
