#include "cosyvoice-internal.h"
#include "cosyvoice-model.h"
#include "cosyvoice-kv-cache.h"
#include "simd-kernels.h"

#include <algorithm>
#include <cstring>
#include <span>
#include <ranges>
#include <vector>

// Layers [0, kInstMaskMaxLayer) mask the instruction body span during INSTRUCT
// generation so alignment heads cannot copy instruction tokens verbatim, while
// upper layers [kInstMaskMaxLayer .. N) remain unmasked for prosody and style.
static constexpr int kInstMaskMaxLayer = 18;

static void build_causal_mask(ggml_fp16_t* mask, uint32_t n_batch, uint32_t seq_len)
{
    uint32_t visible_prefix_end = seq_len - n_batch;
    for (uint32_t i = 0; i != n_batch; ++i)
    {
        for (uint32_t j = 0; j != seq_len; ++j)
            *mask++ = j > visible_prefix_end ? 0xFC00 /* -inf */ : 0;
        ++visible_prefix_end;
    }
}

static ggml_tensor* build_qwen2_decoder_layer(const Qwen2DecoderLayer& layer, ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor* hidden_states, ggml_tensor* position_ids, cosyvoice_kv_cache& kv_cache, ggml_tensor* attention_mask, float rope_theta, float rms_norm_eps, int num_attention_heads, int num_key_value_heads, int layer_idx, bool last_token_only = false, ggml_tensor* last_token_mask = nullptr)
{
    auto residual = hidden_states;
    hidden_states = layer.input_layernorm.build_cgraph(ctx0, hidden_states, rms_norm_eps);

    // For the next-token distribution only the final position is needed, so
    // shrink the query source to the last column up front: K/V still consume
    // the full batch to fill the cache, but the Q projection and everything
    // after the attention run on a single column.
    ggml_tensor* q_source = hidden_states;
    ggml_tensor* q_positions = position_ids;
    if (last_token_only)
    {
        q_source = ggml_view_2d(
            ctx0, hidden_states,
            hidden_states->ne[0], 1,
            hidden_states->nb[1], hidden_states->nb[1] * (hidden_states->ne[1] - 1));
        q_positions = ggml_view_1d(
            ctx0, position_ids, 1,
            position_ids->nb[0] * (position_ids->ne[0] - 1));
        residual = ggml_view_2d(
            ctx0, residual,
            residual->ne[0], 1,
            residual->nb[1], residual->nb[1] * (residual->ne[1] - 1));
        attention_mask = last_token_mask;
    }

    const bool attn_has_lora = (layer.self_attn.q_proj.lora_enabled && *layer.self_attn.q_proj.lora_enabled)
        || (layer.self_attn.v_proj.lora_enabled && *layer.self_attn.v_proj.lora_enabled);
    const bool qkv_bias_ready = (!layer.self_attn.q_proj.bias && !layer.self_attn.k_proj.bias && !layer.self_attn.v_proj.bias)
        || layer.self_attn.qkv_bias != nullptr;

    ggml_tensor* query_states;
    ggml_tensor* key_states;
    ggml_tensor* value_states;
    if (layer.self_attn.qkv_weight && qkv_bias_ready && !last_token_only && !attn_has_lora)
    {
        auto qkv = ggml_mul_mat(ctx0, layer.self_attn.qkv_weight, hidden_states);
        if (layer.self_attn.qkv_bias)
            qkv = ggml_add(ctx0, qkv, layer.self_attn.qkv_bias);

        const int64_t q_dim = layer.self_attn.q_proj.weight->ne[1];
        const int64_t k_dim = layer.self_attn.k_proj.weight->ne[1];
        const int64_t v_dim = layer.self_attn.v_proj.weight->ne[1];
        const int64_t n_tok = qkv->ne[1];
        const int64_t q_hd = q_dim / num_attention_heads;
        const int64_t k_hd = k_dim / num_key_value_heads;
        const int64_t v_hd = v_dim / num_key_value_heads;

        query_states = ggml_view_3d(ctx0, qkv, q_hd, num_attention_heads, n_tok, q_hd * qkv->nb[0], qkv->nb[1], 0);
        key_states   = ggml_view_3d(ctx0, qkv, k_hd, num_key_value_heads, n_tok, k_hd * qkv->nb[0], qkv->nb[1], q_dim * qkv->nb[0]);
        value_states = ggml_view_2d(ctx0, qkv, v_dim, n_tok, qkv->nb[1], (q_dim + k_dim) * qkv->nb[0]);
        if (n_tok > 1)
            value_states = ggml_cont(ctx0, value_states);
        value_states = ggml_reshape_3d(ctx0, value_states, v_hd, num_key_value_heads, n_tok);
    }
    else
    {
        query_states = layer.self_attn.q_proj.build_cgraph(ctx0, q_source);
        key_states = layer.self_attn.k_proj.build_cgraph(ctx0, hidden_states);
        value_states = layer.self_attn.v_proj.build_cgraph(ctx0, hidden_states);

        query_states = ggml_reshape_3d(
            ctx0, query_states,
            query_states->ne[0] / num_attention_heads,
            num_attention_heads,
            query_states->ne[1]);
        key_states = ggml_reshape_3d(
            ctx0, key_states,
            key_states->ne[0] / num_key_value_heads,
            num_key_value_heads,
            key_states->ne[1]);
        value_states = ggml_reshape_3d(
            ctx0, value_states,
            value_states->ne[0] / num_key_value_heads,
            num_key_value_heads,
            value_states->ne[1]);
    }

    query_states = ggml_rope_ext(ctx0, query_states, q_positions, nullptr, static_cast<int>(query_states->ne[0]), GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    key_states = ggml_rope_ext(ctx0, key_states, position_ids, nullptr, static_cast<int>(key_states->ne[0]), GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    kv_cache.update_cache(ctx0, gf, key_states, value_states, position_ids, static_cast<int>(layer_idx));
    query_states = ggml_permute(ctx0, query_states, 0, 2, 1, 3);

    ggml_tensor* attn_output = kv_cache.attention_forward(ctx0, query_states, key_states, value_states, attention_mask);
    attn_output = ggml_reshape_2d(
        ctx0, attn_output,
        attn_output->ne[0] * attn_output->ne[1],
        attn_output->ne[2]);
    attn_output = layer.self_attn.o_proj.build_cgraph(ctx0, attn_output);

    hidden_states = ggml_add(ctx0, residual, attn_output);

    residual = hidden_states;
    hidden_states = layer.post_attention_layernorm.build_cgraph(ctx0, hidden_states, rms_norm_eps);
    hidden_states = layer.mlp.build_cgraph(ctx0, hidden_states);
    hidden_states = ggml_add(ctx0, residual, hidden_states);
    return hidden_states;
}

static void set_graph_backend(ggml_cgraph* gf, ggml_backend_sched_t sched, ggml_backend_t backend, ggml_backend_t cpu_backend, ggml_tensor* cpu_pivot = nullptr)
{
    bool cpu = false;
    for (auto node : ggml_cgraph_node_iterator(gf))
    {
        if (node == cpu_pivot)
            cpu = true;

        // Preserve device acceleration while accumulating LLM matrix products in F32.
        if (node->op == GGML_OP_MUL_MAT)
            ggml_prec_set_acc(node, GGML_PREC_F32);
        ggml_backend_sched_set_tensor_backend(sched, node, cpu ? cpu_backend : backend);
    }
}

static char* ensure_pinned_nucleus(cosyvoice_worker_context* worker, size_t nbytes)
{
    if (worker->pinned_nucleus_buffer && worker->pinned_nucleus_capacity >= nbytes)
        return reinterpret_cast<char*>(ggml_backend_buffer_get_base(worker->pinned_nucleus_buffer.get()));
    if (auto dev = ggml_backend_get_device(worker->backend.get()))
    {
        if (auto host_buft = ggml_backend_dev_host_buffer_type(dev))
        {
            const size_t alloc_bytes = std::max<size_t>(nbytes, 4096);
            if (auto buf = ggml_backend_buft_alloc_buffer(host_buft, alloc_bytes))
            {
                worker->pinned_nucleus_buffer.reset(buf);
                worker->pinned_nucleus_capacity = ggml_backend_buffer_get_size(buf);
                return reinterpret_cast<char*>(ggml_backend_buffer_get_base(buf));
            }
        }
    }
    return nullptr;
}

static size_t pinned_nucleus_probs_offset(int k)
{
    return (static_cast<size_t>(k) * sizeof(int32_t) + 63u) & ~size_t(63u);
}

// Build the logits -> softmax -> top-k -> nucleus-probs tail starting from the
// final hidden-states tensor. Assigns `worker->llm_probs` on success.
static void build_probs_tail(
    cosyvoice_worker_context* worker,
    cosyvoice_model_shared* shared,
    cosyvoice_model_3_shared* cv3_shared,
    ggml_context* ctx0,
    ggml_cgraph* gf,
    ggml_tensor* hidden_states)
{
    const auto& llm = cv3_shared->llm;

    hidden_states = llm.norm.build_cgraph(ctx0, hidden_states, llm.rms_norm_eps);

    auto logits = llm.llm_decoder.build_cgraph(ctx0, hidden_states);
    auto full_probs = ggml_soft_max_ext(ctx0, logits, nullptr, 1.f / worker->config.temperature, 0.f);
    ggml_set_output(full_probs); // Keep the full distribution alive until the host readback.
    worker->llm_full_probs = full_probs;
    worker->llm_top_k = nullptr;

    auto probs = full_probs;
    if (shared->op_caps.top_k)
    {
        const int k = worker->config.sampling.top_k;
        const size_t probs_off = pinned_nucleus_probs_offset(k);
        const size_t total_pinned = probs_off + static_cast<size_t>(k) * sizeof(float);
        const bool split_readback = ensure_pinned_nucleus(worker, total_pinned) != nullptr;

        auto top_k = ggml_top_k(ctx0, probs, k);
        probs = ggml_reshape_2d(ctx0, probs, 1, probs->ne[0]);
        probs = ggml_get_rows(ctx0, probs, top_k);

        if (split_readback)
        {
            ggml_set_output(top_k);
            ggml_set_output(probs);
            ggml_build_forward_expand(gf, probs);
            set_graph_backend(gf, worker->sched.get(), worker->backend.get(), worker->cpu_backend.get());
            worker->llm_top_k = top_k;
        }
        else
        {
            top_k = ggml_reshape_2d(ctx0, top_k, 1, top_k->ne[0]);
            top_k->type = GGML_TYPE_F32;
            probs = ggml_concat(ctx0, top_k, probs, 0);

            ggml_build_forward_expand(gf, probs);
            set_graph_backend(gf, worker->sched.get(), worker->backend.get(), worker->cpu_backend.get());
        }
    }
    else
    {
        auto sched = worker->sched.get();
        auto cpu_backend = worker->cpu_backend.get();

        probs = ggml_dup(ctx0, probs);
        ggml_set_output(probs); // The CPU-copy distribution is also read after graph completion.
        worker->llm_full_probs = probs;
        auto cpu_pivot = probs;
        auto top_k = ggml_top_k(ctx0, probs, worker->config.sampling.top_k);

        probs = ggml_reshape_2d(ctx0, probs, 1, probs->ne[0]);
        probs = ggml_get_rows(ctx0, probs, top_k);
        top_k = ggml_reshape_2d(ctx0, top_k, 1, top_k->ne[0]);
        top_k->type = GGML_TYPE_F32;
        probs = ggml_concat(ctx0, top_k, probs, 0);
        ggml_build_forward_expand(gf, probs);
        set_graph_backend(gf, sched, worker->backend.get(), cpu_backend, cpu_pivot);
    }

    worker->llm_probs = probs;
}

// Copy the computed probs out of the backend into host memory, mirroring the
// layout expected by llm_prepare_probs(). The candidates land in `nucleus_raw`
// and stay untouched: llm_prepare_probs() derives the sampling nucleus from
// this snapshot, so preparing twice for one logits state is a no-op.
static void fetch_llm_probs(
    cosyvoice_worker_context* worker,
    cosyvoice_model_shared* shared,
    cosyvoice_model_3_shared* cv3_shared,
    bool eager_full_probs)
{
    const auto llm_probs = worker->llm_probs;
    GGML_ASSERT(llm_probs);

    auto probs = reinterpret_cast<cosyvoice_llm_token_prob_t*>(worker->nucleus_raw.get());
    auto backend = shared->op_caps.top_k ? worker->backend.get() : worker->cpu_backend.get();
    const bool split_readback = worker->llm_top_k != nullptr;
    int32_t* pinned_ids = nullptr;
    float* pinned_ps = nullptr;
    int k = 0;

    if (split_readback)
    {
        k = static_cast<int>(worker->llm_top_k->ne[0]);
        const size_t probs_off = pinned_nucleus_probs_offset(k);
        auto* pinned_base = ensure_pinned_nucleus(worker, probs_off + static_cast<size_t>(k) * sizeof(float));
        GGML_ASSERT(pinned_base);
        pinned_ids = reinterpret_cast<int32_t*>(pinned_base);
        pinned_ps = reinterpret_cast<float*>(pinned_base + probs_off);
        ggml_backend_tensor_get_async(backend, worker->llm_top_k, pinned_ids, 0, static_cast<size_t>(k) * sizeof(int32_t));
        ggml_backend_tensor_get_async(backend, llm_probs, pinned_ps, 0, static_cast<size_t>(k) * sizeof(float));
    }
    else
    {
        ggml_backend_tensor_get_async(backend, llm_probs, probs, 0, ggml_nbytes(llm_probs));
    }

    if (eager_full_probs)
    {
        ggml_backend_tensor_get_async(
            backend,
            worker->llm_full_probs,
            worker->probs.get(),
            0,
            sizeof(float) * cv3_shared->llm.llm_decoder.weight->ne[1]);
        worker->raw_probs_fetched = true;
    }
    else
    {
        worker->raw_probs_fetched = false;
    }
    cosyvoice_report_progress(worker,7);
    ggml_backend_sched_synchronize(worker->sched.get());
    if (split_readback)
    {
        for (int i = 0; i != k; ++i)
        {
            probs[i].token_id = pinned_ids[i];
            probs[i].prob = pinned_ps[i];
        }
    }
    cosyvoice_report_progress(worker,8);
    worker->raw_probs_masked = false;
    worker->raw_probs_need_stop_mask = false;
}

bool cosyvoice_model_3::llm_prefill(
    ggml_type type,
    const void* data,
    uint32_t n_tokens
)
{
    cosyvoice_progress_scope progress_scope(worker,1);
    auto kv_cache = &worker->llm_kv_cache;
    auto total_len = n_tokens + kv_cache->cur_len;
    if (total_len > shared->params.n_max_seq - 1) return false;
    if (n_tokens > shared->params.n_batch) return false;

    auto causal_mask = n_tokens == 1 ? nullptr : worker->causal_mask;
    if (causal_mask)
    {
        causal_mask->ne[0] = total_len;
        causal_mask->ne[1] = n_tokens;
        causal_mask->nb[1] = total_len * sizeof(ggml_fp16_t);
        causal_mask->nb[3] = causal_mask->nb[2] = causal_mask->ne[1] * causal_mask->nb[1];
    }

    auto& position_ids = worker->position_ids;
    position_ids->ne[0] = n_tokens;
    ggml_backend_tensor_set(position_ids, shared->full_position_ids.get() + kv_cache->cur_len, 0, n_tokens * sizeof(int32_t));

    auto& gf = worker->gf;
    auto& llm_input = worker->llm_input;
    auto& llm_probs = worker->llm_probs;
    if (gf && llm_input && !llm_probs
        && llm_input->type == type && n_tokens == llm_input->ne[1]
        && kv_cache->can_reuse() && !causal_mask)
    {
        ggml_backend_tensor_set(llm_input, data, 0, ggml_nbytes(llm_input));
        kv_cache->shift_kv_node_pos(n_tokens);
    }
    else
    {
        auto ctx0 = worker->ctx0.get();
        auto& llm = cv3_shared->llm;
        ggml_reset(ctx0);
        ggml_backend_sched_reset(worker->sched.get());

        gf = ggml_new_graph(ctx0);
        llm_input = ggml_new_tensor_2d(
            ctx0,
            type,
            llm.speech_embedding_weight->ne[0],
            n_tokens);
        llm_probs = nullptr;
        worker->llm_top_k = nullptr;
        worker->llm_full_probs = nullptr;

        ggml_tensor* emb_pos = nullptr;
        auto hidden_states = llm_input;
        if (type != GGML_TYPE_F32)
        {
            emb_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
            // Written once, then reused with the graph. Keep later intermediates
            // from overwriting these embedding indices in the allocator buffer.
            ggml_set_output(emb_pos);

            hidden_states = ggml_get_rows(ctx0, llm_input, emb_pos);
        }

        auto num_attention_heads = llm.num_attention_heads;
        auto num_key_value_heads = llm.num_key_value_heads;
        for (auto end = static_cast<int>(llm.layers.size() - 1), i = 0; i != end; ++i)
            hidden_states = build_qwen2_decoder_layer(
                llm.layers[i],
                ctx0, gf,
                hidden_states, position_ids, *kv_cache, causal_mask,
                llm.rope_theta,
                llm.rms_norm_eps,
                num_attention_heads, num_key_value_heads,
                i);

        // Final layer.
        const auto& layer = llm.layers.back();

        hidden_states = layer.input_layernorm.build_cgraph(ctx0, hidden_states, llm.rms_norm_eps);

        auto key_states = layer.self_attn.k_proj.build_cgraph(ctx0, hidden_states);
        auto value_states = layer.self_attn.v_proj.build_cgraph(ctx0, hidden_states);

        key_states = ggml_reshape_3d(
            ctx0, key_states,
            key_states->ne[0] / num_key_value_heads,
            num_key_value_heads,
            key_states->ne[1]);
        value_states = ggml_reshape_3d(
            ctx0, value_states,
            value_states->ne[0] / num_key_value_heads,
            num_key_value_heads,
            value_states->ne[1]);

        key_states = ggml_rope_ext(ctx0, key_states, position_ids, nullptr, static_cast<int>(key_states->ne[0]), GGML_ROPE_TYPE_NEOX, 0, llm.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        kv_cache->update_cache(ctx0, gf, key_states, value_states, position_ids, static_cast<int>(llm.layers.size() - 1));

        if (causal_mask)
        {
            causal_mask->buffer = nullptr;
            causal_mask->data = nullptr;
        }
        set_graph_backend(gf, worker->sched.get(), worker->backend.get(), worker->cpu_backend.get());
        ggml_backend_sched_alloc_graph(worker->sched.get(), gf);
        if (causal_mask)
            build_causal_mask(reinterpret_cast<ggml_fp16_t*>(causal_mask->data), n_tokens, total_len);
        if (emb_pos)
            ggml_backend_tensor_set(emb_pos, shared->full_position_ids.get(), 0, n_tokens * sizeof(int32_t));
        kv_cache->set_input_v_idxs(worker->backend.get(), shared->full_position_ids.get() + kv_cache->cur_len, n_tokens, n_tokens);
        ggml_backend_tensor_set(llm_input, data, 0, ggml_nbytes(llm_input));
        kv_cache->cur_len += n_tokens;
    }

    cosyvoice_report_progress(worker,5);
    worker->status = ggml_backend_sched_graph_compute(worker->sched.get(), gf);
    cosyvoice_report_progress(worker,6);
    return worker->status == GGML_STATUS_SUCCESS;
}

bool cosyvoice_model_3::llm_prefill_logits(
    ggml_type type,
    const void* data,
    uint32_t n_tokens
)
{
    if (n_tokens == 1) return llm_decode(type, data);
    cosyvoice_progress_scope progress_scope(worker,1);

    auto kv_cache = &worker->llm_kv_cache;
    auto total_len = n_tokens + kv_cache->cur_len;
    if (total_len > shared->params.n_max_seq - 1) return false;
    if (n_tokens > shared->params.n_batch) return false;

    auto causal_mask = worker->causal_mask;
    causal_mask->ne[0] = total_len;
    causal_mask->ne[1] = n_tokens;
    causal_mask->nb[1] = total_len * sizeof(ggml_fp16_t);
    causal_mask->nb[3] = causal_mask->nb[2] = causal_mask->ne[1] * causal_mask->nb[1];

    const bool need_task_mask = worker->inst_mask_begin < worker->inst_mask_end;
    auto causal_mask_task = worker->causal_mask_task;
    if (need_task_mask)
    {
        causal_mask_task->ne[0] = total_len;
        causal_mask_task->ne[1] = n_tokens;
        causal_mask_task->nb[1] = total_len * sizeof(ggml_fp16_t);
        causal_mask_task->nb[3] = causal_mask_task->nb[2] = causal_mask_task->ne[1] * causal_mask_task->nb[1];

        worker->decode_mask->ne[0] = total_len;
        worker->decode_mask->ne[1] = 1;
        worker->decode_mask->nb[1] = total_len * sizeof(ggml_fp16_t);
        worker->decode_mask->nb[3] = worker->decode_mask->nb[2] = worker->decode_mask->nb[1];
    }

    auto& position_ids = worker->position_ids;
    position_ids->ne[0] = n_tokens;
    ggml_backend_tensor_set(position_ids, shared->full_position_ids.get() + kv_cache->cur_len, 0, n_tokens * sizeof(int32_t));

    auto& gf = worker->gf;
    auto& llm_input = worker->llm_input;
    auto& llm_probs = worker->llm_probs;

    auto ctx0 = worker->ctx0.get();
    auto& llm = cv3_shared->llm;
    ggml_reset(ctx0);
    ggml_backend_sched_reset(worker->sched.get());

    gf = ggml_new_graph(ctx0);
    llm_input = ggml_new_tensor_2d(
        ctx0,
        type,
        llm.speech_embedding_weight->ne[0],
        n_tokens);
    llm_probs = nullptr;
    worker->llm_top_k = nullptr;
    worker->llm_full_probs = nullptr;

    ggml_tensor* emb_pos = nullptr;
    auto hidden_states = llm_input;
    if (type != GGML_TYPE_F32)
    {
        emb_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_output(emb_pos);

        hidden_states = ggml_get_rows(ctx0, llm_input, emb_pos);
    }

    auto num_attention_heads = llm.num_attention_heads;
    auto num_key_value_heads = llm.num_key_value_heads;
    bool used_plain_mask = false;
    bool used_task_mask = false;
    for (auto end = static_cast<int>(llm.layers.size()), i = 0; i != end; ++i)
    {
        const bool layer_masked = need_task_mask && i < kInstMaskMaxLayer;
        if (i + 1 != end)
        {
            if (layer_masked) used_task_mask = true;
            else used_plain_mask = true;
        }
        hidden_states = build_qwen2_decoder_layer(
            llm.layers[i],
            ctx0, gf,
            hidden_states, position_ids, *kv_cache,
            layer_masked ? causal_mask_task : causal_mask,
            llm.rope_theta,
            llm.rms_norm_eps,
            num_attention_heads, num_key_value_heads,
            i, i + 1 == end,
            (layer_masked && i + 1 == end) ? worker->decode_mask : nullptr);
    }

    build_probs_tail(worker, shared, cv3_shared, ctx0, gf, hidden_states);

    causal_mask->buffer = nullptr;
    causal_mask->data = nullptr;
    causal_mask_task->buffer = nullptr;
    causal_mask_task->data = nullptr;
    set_graph_backend(gf, worker->sched.get(), worker->backend.get(), worker->cpu_backend.get());
    ggml_backend_sched_alloc_graph(worker->sched.get(), gf);
    if (used_plain_mask)
        build_causal_mask(reinterpret_cast<ggml_fp16_t*>(causal_mask->data), n_tokens, total_len);
    if (used_task_mask)
    {
        std::vector<ggml_fp16_t> tmp(static_cast<size_t>(n_tokens) * total_len);
        build_causal_mask(tmp.data(), n_tokens, total_len);
        auto* last_row = tmp.data() + static_cast<size_t>(n_tokens - 1) * total_len;
        const uint32_t j_end = std::min<uint32_t>(worker->inst_mask_end, total_len);
        for (uint32_t j = worker->inst_mask_begin; j < j_end; ++j)
            last_row[j] = 0xFC00; // -inf in FP16
        ggml_backend_tensor_set(causal_mask_task, tmp.data(), 0, tmp.size() * sizeof(ggml_fp16_t));
    }
    if (emb_pos)
        ggml_backend_tensor_set(emb_pos, shared->full_position_ids.get(), 0, n_tokens * sizeof(int32_t));
    kv_cache->set_input_v_idxs(worker->backend.get(), shared->full_position_ids.get() + kv_cache->cur_len, n_tokens, n_tokens);
    ggml_backend_tensor_set(llm_input, data, 0, ggml_nbytes(llm_input));
    kv_cache->cur_len += n_tokens;
    cosyvoice_report_progress(worker,5);
    const bool coalesce_readback = shared->op_caps.top_k;
    worker->status = coalesce_readback
        ? ggml_backend_sched_graph_compute_async(worker->sched.get(), gf)
        : ggml_backend_sched_graph_compute(worker->sched.get(), gf);
    cosyvoice_report_progress(worker,6);
    if (worker->status == GGML_STATUS_SUCCESS)
    {
        fetch_llm_probs(worker, shared, cv3_shared, !using_builtin_sampler());
        return true;
    }
    if (coalesce_readback)
        ggml_backend_sched_synchronize(worker->sched.get());
    return false;
}

bool cosyvoice_model_3::llm_decode(ggml_type type, const void* data)
{
    cosyvoice_progress_scope progress_scope(worker,2);
    auto kv_cache = &worker->llm_kv_cache;
    if (kv_cache->cur_len + 1 > shared->params.n_max_seq) return false;

    const auto n_embd = cv3_shared->llm.speech_embedding_weight->ne[0];
    float emb_stack[2048];
    std::vector<float> emb_heap;
    if (type == GGML_TYPE_F16)
    {
        float* dst = n_embd <= 2048 ? emb_stack : (emb_heap.resize(static_cast<size_t>(n_embd)), emb_heap.data());
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(data), dst, n_embd);
        type = GGML_TYPE_F32;
        data = dst;
    }

    auto& position_ids = worker->position_ids;
    position_ids->ne[0] = 1;
    ggml_backend_tensor_set(position_ids, shared->full_position_ids.get() + kv_cache->cur_len, 0, sizeof(int32_t));

    auto& gf = worker->gf;
    auto& llm_input = worker->llm_input;
    auto& llm_probs = worker->llm_probs;
    const bool need_decode_mask = worker->inst_mask_begin < worker->inst_mask_end;
    if (gf && llm_input && llm_probs
        && llm_input->type == type && 1 == llm_input->ne[1]
        && worker->decode_mask_active == need_decode_mask
        && kv_cache->can_reuse())
    {
        ggml_backend_tensor_set(llm_input, data, 0, ggml_nbytes(llm_input));
        kv_cache->shift_kv_node_pos(1);
        if (need_decode_mask)
        {
            worker->decode_mask->ne[0] = kv_cache->cur_len;
            worker->decode_mask->nb[1] = worker->decode_mask->ne[0] * sizeof(ggml_fp16_t);
            worker->decode_mask->nb[3] = worker->decode_mask->nb[2] = worker->decode_mask->nb[1];
        }
    }
    else
    {
        auto ctx0 = worker->ctx0.get();
        auto& llm = cv3_shared->llm;
        ggml_reset(ctx0);
        ggml_backend_sched_reset(worker->sched.get());

        gf = ggml_new_graph(ctx0);
        llm_input = ggml_new_tensor_1d(
            ctx0,
            type,
            llm.speech_embedding_weight->ne[0]);
        llm_probs = nullptr;
        worker->llm_top_k = nullptr;
        worker->llm_full_probs = nullptr;
        worker->decode_mask_active = need_decode_mask;
        if (need_decode_mask)
        {
            worker->decode_mask->ne[0] = kv_cache->cur_len + 1;
            worker->decode_mask->ne[1] = 1;
            worker->decode_mask->nb[1] = worker->decode_mask->ne[0] * sizeof(ggml_fp16_t);
            worker->decode_mask->nb[3] = worker->decode_mask->nb[2] = worker->decode_mask->nb[1];
        }

        ggml_tensor* emb_pos = nullptr;
        auto hidden_states = llm_input;
        if (type != GGML_TYPE_F32)
        {
            emb_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
            ggml_set_output(emb_pos);

            hidden_states = ggml_get_rows(ctx0, llm_input, emb_pos);
        }

        auto num_attention_heads = llm.num_attention_heads;
        auto num_key_value_heads = llm.num_key_value_heads;
        for (auto end = static_cast<int>(llm.layers.size()), i = 0; i != end; ++i)
            hidden_states = build_qwen2_decoder_layer(
                llm.layers[i],
                ctx0, gf,
                hidden_states, position_ids, *kv_cache,
                (need_decode_mask && i < kInstMaskMaxLayer) ? worker->decode_mask : nullptr,
                llm.rope_theta,
                llm.rms_norm_eps,
                num_attention_heads, num_key_value_heads,
                i);

        build_probs_tail(worker, shared, cv3_shared, ctx0, gf, hidden_states);

        ggml_backend_sched_alloc_graph(worker->sched.get(), gf);
        if (emb_pos)
            ggml_backend_tensor_set(emb_pos, shared->full_position_ids.get(), 0, sizeof(int32_t));
        kv_cache->set_input_v_idxs(worker->backend.get(), shared->full_position_ids.get() + kv_cache->cur_len, 1, 1);
        ggml_backend_tensor_set(llm_input, data, 0, ggml_nbytes(llm_input));
        ++kv_cache->cur_len;
    }

    cosyvoice_report_progress(worker,5);
    const bool coalesce_readback = shared->op_caps.top_k;
    worker->status = coalesce_readback
        ? ggml_backend_sched_graph_compute_async(worker->sched.get(), gf)
        : ggml_backend_sched_graph_compute(worker->sched.get(), gf);
    cosyvoice_report_progress(worker,6);
    if (worker->status == GGML_STATUS_SUCCESS)
    {
        fetch_llm_probs(worker, shared, cv3_shared, !using_builtin_sampler());
        return true;
    }
    if (coalesce_readback)
        ggml_backend_sched_synchronize(worker->sched.get());
    return false;
}



void cosyvoice_model_3::llm_prepare_probs(bool allow_stop_tokens)
{
    GGML_ASSERT(worker->llm_probs);
    GGML_ASSERT(worker->nucleus_raw);

    // Always rebuild the nucleus from the immutable top-k snapshot of the last
    // computed logits. Filtering the already normalized nucleus in place would
    // re-apply top_p to values that sum to one, collapsing the distribution to a
    // single candidate whenever the first probability exceeds top_p.
    auto probs = reinterpret_cast<cosyvoice_llm_token_prob_t*>(worker->nucleus_probs.get());
    const auto raw_probs = reinterpret_cast<const cosyvoice_llm_token_prob_t*>(worker->nucleus_raw.get());
    int k = worker->config.sampling.top_k;
    memcpy(probs, raw_probs, static_cast<size_t>(k) * sizeof(*probs));


    const float top_p = worker->config.sampling.top_p;
    std::sort(probs, probs + k, [](const cosyvoice_llm_token_prob_t& a, const cosyvoice_llm_token_prob_t& b)
        { return a.prob > b.prob; });

    float p = 0.f;
    for (int i = 0; i != k; ++i)
    {
        if (!allow_stop_tokens && llm_is_stop_token(probs[i].token_id))
            probs[i].prob = 0.f;
        p += probs[i].prob;
        if (p >= top_p)
        {
            k = i + 1;
            break;
        }
    }

    if (p <= 0.f)
    {
        // Every candidate was masked (e.g. only stop tokens were available); keep
        // an empty nucleus so sampling fails instead of dividing by zero.
        worker->nucleus_probs_len = 0;
        return;
    }


    worker->nucleus_probs_len = k;
    for (int i = 0; i != k; ++i)
        probs[i].prob /= p;

    worker->raw_probs_need_stop_mask = !allow_stop_tokens;
    if (worker->raw_probs_fetched && !allow_stop_tokens && !worker->raw_probs_masked)
    {
        auto vocab_probs = worker->probs.get();
        auto vocab_size = static_cast<uint32_t>(get_speech_token_embed_weight()->ne[1]);
        for (auto token_id : cv3_shared->stop_tokens)
            vocab_probs[token_id] = 0;

        const float sum = simd_dispatch<sum_kernel>(vocab_probs, vocab_size);
        simd_dispatch<div_kernel>(vocab_probs, vocab_size, sum);
        worker->raw_probs_masked = true;
    }
}

void cosyvoice_model_3::ensure_full_probs_ready()
{
    if (!worker->raw_probs_fetched)
    {
        GGML_ASSERT(worker->llm_full_probs);
        auto backend = shared->op_caps.top_k ? worker->backend.get() : worker->cpu_backend.get();
        ggml_backend_tensor_get_async(
            backend,
            worker->llm_full_probs,
            worker->probs.get(),
            0,
            sizeof(float) * cv3_shared->llm.llm_decoder.weight->ne[1]);
        ggml_backend_synchronize(backend);
        worker->raw_probs_fetched = true;
    }
    if (worker->raw_probs_need_stop_mask && !worker->raw_probs_masked)
    {
        auto vocab_probs = worker->probs.get();
        auto vocab_size = static_cast<uint32_t>(get_speech_token_embed_weight()->ne[1]);
        for (auto token_id : cv3_shared->stop_tokens)
            vocab_probs[token_id] = 0;

        const float sum = simd_dispatch<sum_kernel>(vocab_probs, vocab_size);
        simd_dispatch<div_kernel>(vocab_probs, vocab_size, sum);
        worker->raw_probs_masked = true;
    }
}

int cosyvoice_model_3::llm_sample_token()
{
    GGML_ASSERT(worker->llm_probs);
    if (!using_builtin_sampler())
    {
        ensure_full_probs_ready();
        return cosyvoice_model::llm_sample_token();
    }

    auto* nucleus_probs = reinterpret_cast<cosyvoice_llm_token_prob_t*>(worker->nucleus_probs.get());
    const int k = worker->nucleus_probs_len;
    const auto* sampling_params = &worker->config.sampling;
    const auto* accepted_tokens = worker->tokens.data();
    const uint32_t n_accepted_tokens = static_cast<uint32_t>(worker->tokens.size());

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float fallback_random = dist(worker->sampler_rng);
    float nucleus_random = fallback_random;
    for (int i = 0; i != k; ++i)
    {
        nucleus_random -= nucleus_probs[i].prob;
        if (nucleus_random <= 0.0f)
        {
            const int sampled_token = nucleus_probs[i].token_id;
            int repeat_count = 0;
            for (auto t : std::ranges::views::reverse(std::span(accepted_tokens, n_accepted_tokens)) | std::ranges::views::take(sampling_params->win_size))
                if (t == sampled_token)
                    ++repeat_count;
            if (repeat_count < sampling_params->win_size * sampling_params->tau_r)
                return sampled_token;

            ensure_full_probs_ready();
            float* probs = worker->probs.get();
            const uint32_t size = static_cast<uint32_t>(get_speech_token_embed_weight()->ne[1]);
            for (uint32_t j = 0; j != size; ++j)
            {
                fallback_random -= probs[j];
                if (fallback_random <= 0.0f)
                    return static_cast<int>(j);
            }
            break;
        }
    }
    return -1;
}

int cosyvoice_llm_sampler(
    cosyvoice_llm_token_prob_t* nucleus_probs,
    int k,
    float* probs,
    uint32_t size,
    const cosyvoice_sampling_params_t* sampling_params,
    int* accepted_tokens,
    uint32_t n_accepted_tokens,
    cosyvoice_worker_context* workers,
    uint32_t worker_no)
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float fallback_random = dist(workers[worker_no].sampler_rng);
    float nucleus_random = fallback_random;
    for (int i = 0; i != k; ++i)
    {
        nucleus_random -= nucleus_probs[i].prob;
        if (nucleus_random <= 0.0f)
        {
            int sampled_token = nucleus_probs[i].token_id;
            int repeat_count = 0;
            for (auto t : std::ranges::views::reverse(std::span(accepted_tokens, n_accepted_tokens)) | std::ranges::views::take(sampling_params->win_size))
                if (t == sampled_token)
                    ++repeat_count;
            if (repeat_count < sampling_params->win_size * sampling_params->tau_r) return sampled_token;

            for (i = 0; i != size; ++i)
            {
                fallback_random -= probs[i];
                if (fallback_random <= 0.0f)
                    return i;
            }
            break;
        }
    }
    return -1;
}
