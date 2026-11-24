#include "cosyvoice-diagnostics.h"
#include "cosyvoice-internal.h"
#include "cosyvoice-model.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <exception>
#include <random>
#include <stdexcept>
#include <vector>

bool cosyvoice_model_3::llm_job(const int* text, uint32_t text_len, cosyvoice_prompt_t prompt)
{
    llm_clear_accepted_tokens();
    return llm_job_ext(text, text_len, prompt, UINT32_MAX, nullptr);
}

bool cosyvoice_model_3::llm_job_ext(const int* text, uint32_t text_len, cosyvoice_prompt_t prompt, uint32_t max_new_tokens, bool* final)
{
    use_count_guard _guard(this);
    aitable_tts_diag::scope diagnostic(worker->diagnostic, "llm_begin", "llm_end", get_worker_no());
    cosyvoice_progress_scope progress_scope(worker,4);
    const auto& params = shared->params;
    auto& sched = worker->sched;
    const auto& llm = cv3_shared->llm;
    auto& batch_buffer = worker->batch_buffer;
    auto& prompt_crc32 = worker->prompt_crc32;
    auto last_prompt_crc32 = prompt_crc32;
    const bool resuming_stream = llm_get_n_accepted_tokens() > 0;
    bool prompt_prefilled = false;
    bool stop_reached = false;

    if (params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED)
    {
        ggml_backend_sched_reset(sched.get());
        worker->llm_kv_cache.load_cache(worker->backend.get(), sched.get());
    }

    try
    {
        // Isolate the instruction body on lower/mid decoder layers (L0..17) during
        // explicit INSTRUCT generation so alignment heads cannot copy instruction tokens
        // verbatim into speech while L18..23 preserve prosody, emotion, speed, and dialect.
        // ZERO_SHOT, CROSS_LINGUAL, and active ASMR LoRA remain unmasked.
        worker->inst_mask_begin = 0;
        worker->inst_mask_end = 0;
        if (prompt && prompt->mode == COSYVOICE_INFERENCE_MODE_INSTRUCT
            && prompt->llm_prompt_speech_tokens.second == 0
            && !cv3_shared->lora_enabled
            && !prompt->prompt_text.empty())
        {
            const uint32_t p_len = static_cast<uint32_t>(prompt->prompt_text.size());
            uint32_t prefix_toks = 0;
            const auto& pfx = shared->instruction_prefix_tokens;
            if (!pfx.empty() && p_len >= pfx.size()
                && std::equal(pfx.begin(), pfx.end(), prompt->prompt_text.begin()))
                prefix_toks = static_cast<uint32_t>(pfx.size());
            const bool has_eop = shared->endofprompt_token_id >= 0
                && prompt->prompt_text.back() == shared->endofprompt_token_id;
            const uint32_t begin_pos = 1u + prefix_toks;
            // Keep <|endofprompt|> at index p_len visible when present; for raw low-level
            // prompts without a trailing <|endofprompt|>, mask through the final token p_len.
            const uint32_t end_pos = has_eop ? p_len : (p_len + 1u);
            if (begin_pos < end_pos && end_pos <= params.n_max_seq)
            {
                worker->inst_mask_begin = begin_pos;
                worker->inst_mask_end = end_pos;
                worker->decode_mask->ne[0] = params.n_max_seq;
                worker->decode_mask->ne[1] = 1;
                worker->decode_mask->nb[1] = params.n_max_seq * sizeof(ggml_fp16_t);
                worker->decode_mask->nb[3] = worker->decode_mask->nb[2] = worker->decode_mask->nb[1];
                std::vector<ggml_fp16_t> host_mask(params.n_max_seq, 0);
                for (uint32_t j = begin_pos; j < end_pos; ++j)
                    host_mask[j] = 0xFC00; // -inf in FP16
                ggml_backend_tensor_set(worker->decode_mask, host_mask.data(), 0, host_mask.size() * sizeof(ggml_fp16_t));
            }
        }

        const auto n_batch = params.n_batch;
        const auto speech_type = llm.speech_embedding_weight->type;
        const auto speech_row_size = static_cast<uint32_t>(llm.speech_embedding_weight->nb[1]);
        const auto speech_emb = reinterpret_cast<const char*>(llm.speech_embedding_weight->data);
        const auto token_emb = reinterpret_cast<const char*>(llm.embed_tokens_weight->data);
        const char* cur;
        uint32_t offset = 0;

        if (llm_get_n_accepted_tokens() == 0 || llm_get_kv_cache_len() == 0)
        {
            GGML_ASSERT(text);

            const auto token_type = llm.embed_tokens_weight->type;
            const auto token_row_size = static_cast<uint32_t>(llm.embed_tokens_weight->nb[1]);

            auto prefill_embedding = [&](const char* data, int token_id, uint32_t row_size, ggml_type type)
            {
                memcpy(batch_buffer.get() + offset++ * row_size, data + token_id * row_size, row_size);

                if (offset == n_batch)
                {
                    if (!llm_prefill(type, batch_buffer.get(), n_batch))
                        throw std::runtime_error("Failed to prefill LLM KV cache.\n");
                    offset = 0;
                }
            };

            if (llm_get_kv_cache_len() == 0)
            {
                if (speech_type == token_type)
                    prefill_embedding(speech_emb, llm.sos_token_id, speech_row_size, speech_type);
                else
                    llm_prefill(speech_type, speech_emb + llm.sos_token_id * speech_row_size, 1);
            prefill_prompt:
                prompt_prefilled = true;
                for (const auto& i : prompt->prompt_text)
                    prefill_embedding(token_emb, i, token_row_size, token_type);
                prompt_crc32 = prompt->prompt_crc32;

                if (params.strict_seed_mode && offset != 0)
                {
                    if (!llm_prefill(token_type, batch_buffer.get(), offset))
                        throw std::runtime_error("Failed to prefill LLM KV cache.\n");
                    offset = 0;
                }
            }
            else if (prompt_crc32 != prompt->prompt_crc32
                || !llm_set_kv_cache_len(1 + static_cast<uint32_t>(prompt->prompt_text.size())))
            {
                // The first token is assumed to be the SOS token already stored in the KV cache.
                llm_set_kv_cache_len(1);
                goto prefill_prompt;
            }

            for (uint32_t i = 0; i != text_len; ++i)
                prefill_embedding(token_emb, text[i], token_row_size, token_type);

            if (token_type != speech_type)
            {
                if (offset != 0 && !llm_prefill(token_type, batch_buffer.get(), offset))
                    throw std::runtime_error("Failed to prefill LLM KV cache.\n");
                offset = 0;
            }

            if (prompt->llm_prompt_speech_tokens.second != 0)
            {
                prefill_embedding(speech_emb, llm.task_token_id, speech_row_size, speech_type);

                const auto end = prompt->llm_prompt_speech_tokens.second - 1;
                for (uint32_t i = 0; i != end; ++i)
                    prefill_embedding(speech_emb, prompt->llm_prompt_speech_tokens.first[i], speech_row_size, speech_type);
                cur = speech_emb + prompt->llm_prompt_speech_tokens.first[end] * speech_row_size;
            }
            else cur = speech_emb + llm.task_token_id * speech_row_size;

            if (const auto n_acc = llm_get_n_accepted_tokens(); n_acc > 0)
            {
                const auto* acc_tokens = llm_get_accepted_tokens();
                const auto end = n_acc - 1;
                prefill_embedding(cur, 0, speech_row_size, speech_type);
                for (uint32_t i = 0; i < end; ++i)
                    prefill_embedding(speech_emb, acc_tokens[i], speech_row_size, speech_type);
                cur = speech_emb + acc_tokens[end] * speech_row_size;
            }

            // Prefill the remaining tokens together with `cur` (the input of
            // the first decode step) in a single fused pass that also computes
            // the logits of the next token.
            memcpy(batch_buffer.get() + offset * speech_row_size, cur, speech_row_size);
            if (!llm_prefill_logits(speech_type, batch_buffer.get(), offset + 1))
                throw std::runtime_error("Failed to prefill LLM KV cache and compute logits.\n");
        }
        else
        {
            // Continue from existing KV cache — get last accepted token's embedding
            const auto last_token_id = llm_get_accepted_tokens()[llm_get_n_accepted_tokens() - 1];
            cur = speech_emb + last_token_id * speech_row_size;
            if (!llm_decode(speech_type, cur))
                throw std::runtime_error("Failed to decode LLM output.\n");
        }

        // First call: min/max from text input
        const auto min_len = static_cast<uint32_t>(text_len * worker->config.min_token_text_ratio);
        const auto max_len = static_cast<uint32_t>(text_len * worker->config.max_token_text_ratio);
        const auto limit = std::min<uint64_t>(static_cast<uint64_t>(llm_get_n_accepted_tokens()) + max_new_tokens, max_len);


        for (uint32_t n = llm_get_n_accepted_tokens();; ++n)
        {
            if (is_stop_requested())
            {
                worker->llm_input = nullptr;
                worker->inst_mask_begin = 0;
                worker->inst_mask_end = 0;
                llm_set_kv_cache_len(0);
                prompt_crc32 = 0;
                if (params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED)
                    worker->llm_kv_cache.clear_offloaded_cache();
                cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_INFO, "LLM generation stopped by user.\n");
                if (params.builtin_sampler_rng_policy == COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION)
                    reset_builtin_sampler_rng();
                return false;
            }

            if (n >= max_len)
            {
                // Do not report truncated synthesis as an EOS completion.
                if (worker->progress_callback) worker->failure_stats.reason = 2;
                throw std::runtime_error("LLM generated token limit reached before stop token.\n");
            }

            if (worker->progress_callback) ++worker->progress.samples;
            cosyvoice_report_progress(worker,2,3);
            // Every yielded token advances generated history and KV, including
            // silence omitted from the vocoder, matching the upstream generator.
            llm_prepare_probs(n > min_len);
            const int token_id = llm_sample_token();
            if (token_id == -1)
                throw std::runtime_error("Failed to sample token from LLM output. This might be wrong with the model or caused by an issue with the sampling parameters.\n");
            if (n > min_len && llm_is_stop_token(token_id))
            {
                stop_reached = true;
                break;
            }


            llm_accept_token(token_id);
            cosyvoice_report_progress(worker,2,3);

            if (is_stop_requested())
            {
                worker->llm_input = nullptr;
                worker->inst_mask_begin = 0;
                worker->inst_mask_end = 0;
                llm_set_kv_cache_len(0);
                prompt_crc32 = 0;
                if (params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED)
                    worker->llm_kv_cache.clear_offloaded_cache();
                cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_INFO, "LLM generation stopped by user.\n");
                if (params.builtin_sampler_rng_policy == COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION)
                    reset_builtin_sampler_rng();
                return false;
            }

            if (n > limit)
                break;

            cur = speech_emb + token_id * speech_row_size;
            if (!llm_decode(speech_type, cur))
                throw std::runtime_error("Failed to decode LLM output.\n");
        }
    }
    catch (const std::exception& e)
    {
        worker->llm_input = nullptr;
        worker->inst_mask_begin = 0;
        worker->inst_mask_end = 0;
        llm_set_kv_cache_len(0);
        prompt_crc32 = 0;
        if (params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED)
            worker->llm_kv_cache.clear_offloaded_cache();
        cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_ERROR, e.what());
        if (params.builtin_sampler_rng_policy == COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION)
            reset_builtin_sampler_rng();
        return false;
    }

    worker->llm_input = nullptr;
    if (final)
        *final = stop_reached;

    if (params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED)
    {
        if (max_new_tokens != UINT32_MAX && !stop_reached)
        {
            llm_offload_kv_cache();
        }
        else if (prompt != nullptr && (prompt_prefilled || resuming_stream || last_prompt_crc32 != prompt_crc32))
        {
            llm_set_kv_cache_len(1 + static_cast<uint32_t>(prompt->prompt_text.size()));
            llm_offload_kv_cache();
        }
    }

    // Reset RNG on stop (matches non-streaming behavior)
    if (stop_reached && params.builtin_sampler_rng_policy == COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION)
        reset_builtin_sampler_rng();

    return true;
}

void cosyvoice_model_3::llm_accept_token(int token)
{
    worker->tokens.push_back(token);
    worker->output_tokens.accept(token, cv3_shared->silent_tokens.count(token) != 0);
}
