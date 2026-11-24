// Model-backed regression for INSTRUCT instruction-attention isolation boundaries:
//   1. Explicit mode routing on the same worker: INSTRUCT vs CROSS_LINGUAL with
//      identical prompt tokens (and round-trip switching INSTRUCT <-> CROSS_LINGUAL
//      <-> ZERO_SHOT) must toggle the L0..17 mask cleanly without leaking state.
//   2. Low-level cosyvoice_prompt_set_ext() with and without trailing <|endofprompt|>:
//      - [prefix, single_inst_token] (no <|endofprompt|>) masks the trailing token;
//      - [single_inst_token] (no prefix, no <|endofprompt|>) masks the trailing token;
//      - [prefix, <|endofprompt|>] (empty body with <|endofprompt|>) stays unmasked;
//      - multi-token prompts with/without <|endofprompt|> produce finite distributions.
//   3. Initial device weight buffer sizing with n_max_seq=4096 and n_workers=2 accounts
//      for per-worker decode_mask without triggering extra spill buffers on large-VRAM GPUs.
#include "cosyvoice.h"
#include "cosyvoice-lowlevel.h"
#include "tool_common.h"
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace
{
struct step_capture
{
    std::vector<std::vector<float>> probs;
    static constexpr int kFixedTokens[5] = { 100, 101, 102, 103, 104 };
};

int deterministic_capture_sampler(
    cosyvoice_llm_token_prob_t*,
    int,
    float* probs,
    uint32_t size,
    const cosyvoice_sampling_params_t*,
    int*,
    uint32_t n_accepted,
    void* opaque)
{
    auto& cap = *static_cast<step_capture*>(opaque);
    cap.probs.emplace_back(probs, probs + size);
    const uint32_t idx = n_accepted < 5 ? n_accepted : 4;
    return step_capture::kFixedTokens[idx];
}

std::vector<int> tokenize_str(cosyvoice_tokenizer_context_t tok, const char* text, bool parse_special)
{
    std::unique_ptr<cosyvoice_tokenization_result, decltype(&cosyvoice_tokenization_result_free)>
        res(cosyvoice_tokenization_result_create(), cosyvoice_tokenization_result_free);
    cosyvoice_tokenize(tok, text, res.get(), parse_special);
    const int* data = cosyvoice_tokenization_result_get_tokens(res.get());
    const uint32_t n = cosyvoice_tokenization_result_get_n_tokens(res.get());
    return std::vector<int>(data, data + n);
}

bool run_steps(
    cosyvoice_context_t ctx,
    const std::vector<int>& text_tokens,
    cosyvoice_prompt_t prompt,
    step_capture& out)
{
    out.probs.clear();
    cosyvoice_llm_clear_accepted_tokens(ctx);
    cosyvoice_set_sampler(ctx, deterministic_capture_sampler, &out);
    bool is_final = false;
    if (!cosyvoice_llm_job_ext(ctx, text_tokens.data(), static_cast<uint32_t>(text_tokens.size()), prompt, 3, &is_final))
        return false;
    if (out.probs.size() < 4)
        return false;
    for (const auto& step : out.probs)
    {
        double sum = 0.0;
        for (float v : step)
        {
            if (!std::isfinite(v) || v < 0.0f)
                return false;
            sum += static_cast<double>(v);
        }
        if (std::fabs(sum - 1.0) > 1e-2)
            return false;
    }
    return true;
}

double max_step_diff(const step_capture& a, const step_capture& b)
{
    if (a.probs.size() != b.probs.size())
        return 1e9;
    double best = 0.0;
    for (size_t s = 0; s < a.probs.size(); ++s)
    {
        if (a.probs[s].size() != b.probs[s].size())
            return 1e9;
        for (size_t j = 0; j < a.probs[s].size(); ++j)
            best = std::max(best, std::fabs(static_cast<double>(a.probs[s][j]) - static_cast<double>(b.probs[s][j])));
    }
    return best;
}

struct log_capture
{
    bool saw_multi_buffer_log = false;
};

void capture_log_cb(enum ggml_log_level level, const char* text, void* user_data)
{
    if (text && user_data && std::strstr(text, "weights: packed into ") != nullptr)
        static_cast<log_capture*>(user_data)->saw_multi_buffer_log = true;
    (void)level;
}

bool run_policy_suite(
    const char* model_path,
    const char* backend_name,
    cosyvoice_prompt_speech_t prompt_speech,
    cosyvoice_inference_buffer_policy_t policy,
    const char* policy_name)
{
    auto backend = ggml_backend_init_by_name(backend_name, nullptr);
    if (!backend)
    {
        std::fprintf(stderr, "[%s] failed to init backend %s\n", policy_name, backend_name);
        return false;
    }

    cosyvoice_context_params_v4_cpp params4{};
    cosyvoice_init_default_context_params(&params4);
    params4.llm_use_flash_attn = true;
    params4.llm_kv_cache_type = COSYVOICE_KV_CACHE_TYPE_F16;
    params4.inference_buffer_policy = policy;
    params4.n_max_seq = 512;
    params4.n_batch = 256;
    params4.n_workers = 1;
    params4.strict_seed_mode = true;

    std::unique_ptr<cosyvoice_context, decltype(&cosyvoice_free)> ctx(
        cosyvoice_load_from_file_ext(model_path, &params4, backend, 4),
        cosyvoice_free);
    if (!ctx)
    {
        std::fprintf(stderr, "[%s] failed to load model\n", policy_name);
        return false;
    }

    auto tok = cosyvoice_get_tokenizer(ctx.get());
    const char* prefix_str = cosyvoice_get_instruction_prefix(ctx.get());
    const auto prefix_toks = (prefix_str && *prefix_str) ? tokenize_str(tok, prefix_str, false) : std::vector<int>{};
    const auto eop_toks = tokenize_str(tok, "<|endofprompt|>", true);
    if (eop_toks.size() != 1)
    {
        std::fprintf(stderr, "[%s] expected 1 <|endofprompt|> token, got %zu\n", policy_name, eop_toks.size());
        return false;
    }
    const int eop_id = eop_toks[0];
    const auto body_toks = tokenize_str(tok, " 请用非常清晰的语气缓慢说话。", false);
    const auto single_tok = tokenize_str(tok, "清晰", false);
    const auto text_toks = tokenize_str(tok, "今天天气不错，我们出去走走吧。", true);
    if (body_toks.size() < 2 || single_tok.empty() || text_toks.size() < 4)
    {
        std::fprintf(stderr, "[%s] unexpected tokenization sizes\n", policy_name);
        return false;
    }

    auto make_prompt = [&](cosyvoice_inference_mode_t mode, const std::vector<int>& instr)
    {
        std::unique_ptr<cosyvoice_prompt, decltype(&cosyvoice_prompt_free)>
            p(cosyvoice_prompt_init_from_prompt_speech(ctx.get(), prompt_speech), cosyvoice_prompt_free);
        cosyvoice_prompt_set_ext(ctx.get(), p.get(), mode,
            instr.empty() ? nullptr : instr.data(),
            static_cast<uint32_t>(instr.size()), true);
        return p;
    };

    // 1. Same worker switching INSTRUCT <-> CROSS_LINGUAL <-> ZERO_SHOT with identical prompt tokens.
    std::vector<int> full_with_eop = prefix_toks;
    full_with_eop.insert(full_with_eop.end(), body_toks.begin(), body_toks.end());
    full_with_eop.push_back(eop_id);

    auto p_inst = make_prompt(COSYVOICE_INFERENCE_MODE_INSTRUCT, full_with_eop);
    auto p_xl   = make_prompt(COSYVOICE_INFERENCE_MODE_CROSS_LINGUAL, full_with_eop);
    auto p_zs   = make_prompt(COSYVOICE_INFERENCE_MODE_ZERO_SHOT, {});

    step_capture inst1, xl1, inst2, xl2, zs1, inst3;
    if (!run_steps(ctx.get(), text_toks, p_inst.get(), inst1)
        || !run_steps(ctx.get(), text_toks, p_xl.get(), xl1)
        || !run_steps(ctx.get(), text_toks, p_inst.get(), inst2)
        || !run_steps(ctx.get(), text_toks, p_xl.get(), xl2)
        || !run_steps(ctx.get(), text_toks, p_zs.get(), zs1)
        || !run_steps(ctx.get(), text_toks, p_inst.get(), inst3))
    {
        std::fprintf(stderr, "[%s] mode-switching run_steps failed\n", policy_name);
        return false;
    }

    const double diff_inst_xl = max_step_diff(inst1, xl1);
    const double diff_inst_rt = max_step_diff(inst1, inst2);
    const double diff_xl_rt   = max_step_diff(xl1, xl2);
    const double diff_inst_zs = max_step_diff(inst1, inst3);
    std::printf("[%s] mode_switch: inst_vs_xl=%.6g inst_rt=%.6g xl_rt=%.6g inst_after_zs=%.6g\n",
        policy_name, diff_inst_xl, diff_inst_rt, diff_xl_rt, diff_inst_zs);
    if (!(diff_inst_xl > 1e-3 && diff_inst_rt < 1e-5 && diff_xl_rt < 1e-5 && diff_inst_zs < 1e-5))
    {
        std::fprintf(stderr, "[%s] FAIL mode_switch boundary check\n", policy_name);
        return false;
    }

    // 2. Low-level prompt WITHOUT <|endofprompt|>: [prefix, single_tok] must mask the trailing token.
    //    Under the old end_pos=p_len heuristic, begin_pos==end_pos left this completely unmasked.
    std::vector<int> pfx_single_no_eop = prefix_toks;
    pfx_single_no_eop.push_back(single_tok[0]);
    auto p_inst_no_eop = make_prompt(COSYVOICE_INFERENCE_MODE_INSTRUCT, pfx_single_no_eop);
    auto p_xl_no_eop   = make_prompt(COSYVOICE_INFERENCE_MODE_CROSS_LINGUAL, pfx_single_no_eop);
    step_capture c_inst_no_eop, c_xl_no_eop;
    if (!run_steps(ctx.get(), text_toks, p_inst_no_eop.get(), c_inst_no_eop)
        || !run_steps(ctx.get(), text_toks, p_xl_no_eop.get(), c_xl_no_eop))
    {
        std::fprintf(stderr, "[%s] pfx_single_no_eop run_steps failed\n", policy_name);
        return false;
    }
    const double diff_pfx_single = max_step_diff(c_inst_no_eop, c_xl_no_eop);
    std::printf("[%s] no_eop_pfx_single: inst_vs_xl=%.6g\n", policy_name, diff_pfx_single);
    if (!(diff_pfx_single > 1e-4))
    {
        std::fprintf(stderr, "[%s] FAIL no_eop_pfx_single was not masked\n", policy_name);
        return false;
    }

    // 3. Raw 1-token prompt without prefix and without <|endofprompt|>: [single_tok].
    std::vector<int> raw_single = { single_tok[0] };
    auto p_inst_raw1 = make_prompt(COSYVOICE_INFERENCE_MODE_INSTRUCT, raw_single);
    auto p_xl_raw1   = make_prompt(COSYVOICE_INFERENCE_MODE_CROSS_LINGUAL, raw_single);
    step_capture c_inst_raw1, c_xl_raw1;
    if (!run_steps(ctx.get(), text_toks, p_inst_raw1.get(), c_inst_raw1)
        || !run_steps(ctx.get(), text_toks, p_xl_raw1.get(), c_xl_raw1))
    {
        std::fprintf(stderr, "[%s] raw_single run_steps failed\n", policy_name);
        return false;
    }
    const double diff_raw1 = max_step_diff(c_inst_raw1, c_xl_raw1);
    std::printf("[%s] no_eop_raw_single: inst_vs_xl=%.6g\n", policy_name, diff_raw1);
    if (!(diff_raw1 > 1e-4))
    {
        std::fprintf(stderr, "[%s] FAIL no_eop_raw_single was not masked\n", policy_name);
        return false;
    }

    // 4. Empty instruction WITH <|endofprompt|>: [prefix, <|endofprompt|>] must remain unmasked.
    std::vector<int> empty_with_eop = prefix_toks;
    empty_with_eop.push_back(eop_id);
    auto p_inst_empty = make_prompt(COSYVOICE_INFERENCE_MODE_INSTRUCT, empty_with_eop);
    auto p_xl_empty   = make_prompt(COSYVOICE_INFERENCE_MODE_CROSS_LINGUAL, empty_with_eop);
    step_capture c_inst_empty, c_xl_empty;
    if (!run_steps(ctx.get(), text_toks, p_inst_empty.get(), c_inst_empty)
        || !run_steps(ctx.get(), text_toks, p_xl_empty.get(), c_xl_empty))
    {
        std::fprintf(stderr, "[%s] empty_with_eop run_steps failed\n", policy_name);
        return false;
    }
    const double diff_empty_eop = max_step_diff(c_inst_empty, c_xl_empty);
    std::printf("[%s] empty_with_eop: inst_vs_xl=%.6g\n", policy_name, diff_empty_eop);
    if (!(diff_empty_eop < 1e-5))
    {
        std::fprintf(stderr, "[%s] FAIL empty_with_eop unexpectedly masked <|endofprompt|>\n", policy_name);
        return false;
    }

    // 5. Multi-token prompt without <|endofprompt|> vs with <|endofprompt|>.
    std::vector<int> full_no_eop = prefix_toks;
    full_no_eop.insert(full_no_eop.end(), body_toks.begin(), body_toks.end());
    auto p_inst_multi_no_eop = make_prompt(COSYVOICE_INFERENCE_MODE_INSTRUCT, full_no_eop);
    auto p_xl_multi_no_eop   = make_prompt(COSYVOICE_INFERENCE_MODE_CROSS_LINGUAL, full_no_eop);
    step_capture c_inst_mno, c_xl_mno;
    if (!run_steps(ctx.get(), text_toks, p_inst_multi_no_eop.get(), c_inst_mno)
        || !run_steps(ctx.get(), text_toks, p_xl_multi_no_eop.get(), c_xl_mno))
    {
        std::fprintf(stderr, "[%s] multi_no_eop run_steps failed\n", policy_name);
        return false;
    }
    const double diff_multi_no_eop = max_step_diff(c_inst_mno, c_xl_mno);
    std::printf("[%s] no_eop_multi: inst_vs_xl=%.6g\n", policy_name, diff_multi_no_eop);
    if (!(diff_multi_no_eop > 1e-3))
    {
        std::fprintf(stderr, "[%s] FAIL no_eop_multi was not masked\n", policy_name);
        return false;
    }

    return true;
}

bool check_device_buffer_budget(const char* model_path, const char* backend_name)
{
    auto backend = ggml_backend_init_by_name(backend_name, nullptr);
    if (!backend)
        return false;

    const size_t max_buf = ggml_backend_get_max_size(backend);
    cosyvoice_context_params_v4_cpp params4{};
    cosyvoice_init_default_context_params(&params4);
    params4.n_max_seq = 4096;
    params4.n_workers = 2;
    params4.inference_buffer_policy = COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED;
    params4.strict_seed_mode = true;

    log_capture log_cap;
    ggml_log_set(capture_log_cb, &log_cap);
    std::unique_ptr<cosyvoice_context, decltype(&cosyvoice_free)> ctx(
        cosyvoice_load_from_file_ext(model_path, &params4, backend, 4),
        cosyvoice_free);
    ggml_log_set(cosyvoice_log_callback_default, nullptr);
    if (!ctx)
    {
        std::fprintf(stderr, "[budget] failed to load model with n_max_seq=4096 n_workers=2\n");
        return false;
    }
    const bool expect_single_buf = max_buf >= (2ull * 1024ull * 1024ull * 1024ull);
    std::printf("[budget] n_max_seq=4096 n_workers=2 max_buf_mib=%zu multi_buffer_split=%s\n",
        max_buf >> 20, log_cap.saw_multi_buffer_log ? "yes" : "no");
    if (expect_single_buf && log_cap.saw_multi_buffer_log)
    {
        std::fprintf(stderr, "[budget] FAIL unexpected multi-buffer spill on large-VRAM backend\n");
        return false;
    }
    return true;
}
} // namespace

int tool_entry(int argc, char** argv)
{
    if (argc != 4)
    {
        std::fprintf(stderr, "Usage: %s MODEL.gguf BACKEND PROMPT_SPEECH.gguf\n", argv[0]);
        return 2;
    }

    cosyvoice_init_backend_from_path(nullptr);
    std::unique_ptr<cosyvoice_prompt_speech, decltype(&cosyvoice_prompt_speech_free)>
        prompt_speech(cosyvoice_prompt_speech_load_from_file(argv[3]), cosyvoice_prompt_speech_free);
    if (!prompt_speech)
    {
        std::fprintf(stderr, "Failed to load prompt speech: %s\n", argv[3]);
        return 2;
    }

    bool ok = true;
    ok = ok && run_policy_suite(argv[1], argv[2], prompt_speech.get(),
        COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED, "balanced");
    ok = ok && run_policy_suite(argv[1], argv[2], prompt_speech.get(),
        COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED, "dedicated");
    ok = ok && check_device_buffer_budget(argv[1], argv[2]);
    std::printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
