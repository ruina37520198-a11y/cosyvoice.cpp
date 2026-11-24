#include "cosyvoice-model.h"
#include "cosyvoice-loader.h"
#include "cosyvoice-kv-cache.h"

#include <tuple>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <format>
#include <span>
#include <ranges>
#include <unordered_map>
#include <unordered_set>

namespace
{
#if defined(__APPLE__) && defined(__aarch64__)
constexpr bool backend_looks_uma(ggml_backend_t backend, ggml_backend_buffer* buffer) { return true; }
#else
constexpr size_t kUmaProbeBytes = 128ull * 1024ull * 1024ull;
constexpr size_t kUmaProbeMinBytes = 64ull * 1024ull * 1024ull;
constexpr int kUmaProbeIters = 4;

struct transfer_fit
{
    double bandwidth_bytes_per_us = 0.0;
    double overhead_us = 0.0;
};

static transfer_fit fit_transfer(size_t size_1, double time_1_us, size_t size_2, double time_2_us)
{
    transfer_fit result;
    if (size_1 <= size_2 || time_1_us <= time_2_us)
        return result;

    const auto bw = static_cast<double>(size_1 - size_2) / (time_1_us - time_2_us);
    if (bw <= 0.0)
        return result;

    result.bandwidth_bytes_per_us = bw;
    result.overhead_us = time_1_us - static_cast<double>(size_1) / bw;
    if (result.overhead_us < 0.0)
        result.overhead_us = 0.0;
    return result;
}

template <typename F>
static double measure_min_us(F&& fn)
{
    double min_us = (std::numeric_limits<double>::max)();
    for (int i = 0; i != kUmaProbeIters; ++i) {
        const auto start = ggml_time_us();
        fn();
        const auto elapsed = ggml_time_us() - start;
        if (elapsed > 0 && static_cast<double>(elapsed) < min_us)
            min_us = static_cast<double>(elapsed);
    }
    return min_us < (std::numeric_limits<double>::max)() ? min_us : 0.0;
}

bool backend_looks_uma(ggml_backend_t backend, ggml_backend_buffer* buffer)
{
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(ggml_backend_get_device(backend), &props);
    if (props.type == GGML_BACKEND_DEVICE_TYPE_IGPU || props.type == GGML_BACKEND_DEVICE_TYPE_CPU)
        return true;
    if (strncmp(props.name, "Vulkan", 6) == 0 && props.type == GGML_BACKEND_DEVICE_TYPE_GPU)
        return false;

    constexpr auto size_1 = kUmaProbeBytes - (kUmaProbeBytes % sizeof(float));
    constexpr auto size_2 = kUmaProbeMinBytes - (kUmaProbeMinBytes % sizeof(float));
    if constexpr (size_1 == 0 || size_2 == 0 || size_1 <= size_2)
        return false;

    if (!backend || !buffer)
        return false;

    const auto buffer_size = ggml_backend_buffer_get_size(buffer);
    if (buffer_size < kUmaProbeBytes)
    {
        cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_INFO, std::format(
            "UMA probe: skipped (buffer of {} bytes is smaller than the {} byte probe)\n",
            buffer_size, kUmaProbeBytes).c_str());
        return false;
    }

    ggml_tensor tensor =
    {
        .type = GGML_TYPE_I8,
        .ne = { static_cast<int64_t>(size_1), 1, 1, 1 },
        .nb = { 1, size_1, size_1, size_1 }
    };
    ggml_backend_tensor_alloc(buffer, &tensor, ggml_backend_buffer_get_base(buffer));
    auto host_src = std::make_unique<char[]>(size_1 * 2);

    auto* src = host_src.get();
    auto* dst = src + size_1;

    memcpy(dst, src, size_1);

    const auto memcpy_1_us = measure_min_us([&] { memcpy(dst, src, size_1); });
    const auto memcpy_2_us = measure_min_us([&] { memcpy(dst, src, size_2); });

    ggml_backend_tensor_set(&tensor, src, 0, size_1);
    const auto backend_1_us = measure_min_us([&] { ggml_backend_tensor_set(&tensor, src, 0, size_1); });

    ggml_backend_tensor_set(&tensor, src, 0, size_2);
    const auto backend_2_us = measure_min_us([&] { ggml_backend_tensor_set(&tensor, src, 0, size_2); });

    if (memcpy_1_us <= 0.0 || memcpy_2_us <= 0.0 || backend_1_us <= 0.0 || backend_2_us <= 0.0)
        return false;

    const auto memcpy_fit = fit_transfer(size_1, memcpy_1_us, size_2, memcpy_2_us);
    const auto backend_fit = fit_transfer(size_1, backend_1_us, size_2, backend_2_us);
    if (memcpy_fit.bandwidth_bytes_per_us <= 0.0 || backend_fit.bandwidth_bytes_per_us <= 0.0)
        return false;

    constexpr double mib_per_us_to_mib_per_s = 1000000.0 / (1024.0 * 1024.0);
    const auto memcpy_mib_s = memcpy_fit.bandwidth_bytes_per_us * mib_per_us_to_mib_per_s;
    const auto backend_mib_s = backend_fit.bandwidth_bytes_per_us * mib_per_us_to_mib_per_s;
    const auto uma = backend_fit.bandwidth_bytes_per_us >= memcpy_fit.bandwidth_bytes_per_us * 0.7;
    cosyvoice_call_ggml_log_callback(
        GGML_LOG_LEVEL_INFO,
        std::format(
            "UMA probe: memcpy={:.1f} MiB/s (overhead {:.1f} us), backend={:.1f} MiB/s (overhead {:.1f} us), guess: UMA={}\n",
            memcpy_mib_s,
            memcpy_fit.overhead_us,
            backend_mib_s,
            backend_fit.overhead_us,
            uma
        ).c_str()
    );

    return uma;
}
#endif
}

#define LOAD_SUBMODULE_EX(name, module) do {\
    auto& _module = module;\
    _module.OnLoad(loader, combine_prefix(prefix, name));\
} while (false)
#define LOAD_SUBMODULE(name) LOAD_SUBMODULE_EX(#name, name)

#define LOAD_TENSOR_EX(name, obj) do {\
    this->obj = loader.get_gguf_tensor(prefix, name);\
    loader.register_tensor(prefix, name, &this->obj);\
} while (false)
#define LOAD_TENSOR(name) LOAD_TENSOR_EX(#name, name)

#define LOAD_OPTIONAL_TENSOR_EX(name, obj) do {\
    this->obj = loader.get_gguf_tensor(prefix, name, true);\
    if (this->obj)\
        loader.register_tensor(prefix, name, &this->obj);\
} while (false)
#define LOAD_OPTIONAL_TENSOR(name) LOAD_OPTIONAL_TENSOR_EX(#name, name)

#define LOAD_METADATA(name) GGML_ASSERT(loader.get_metadata(prefix, #name, name))
#define LOAD_METADATA_NOPREFIX(name) GGML_ASSERT(loader.get_metadata(#name, name))

constexpr uint32_t GGML_TENSOR_FLAG_NEED_EPS_ADJUST = 28u << 0;

static inline void set_tensor_eps(ggml_tensor* tensor, float eps)
{
    GGML_ASSERT(tensor && tensor->type == GGML_TYPE_F32);

    tensor->flags |= GGML_TENSOR_FLAG_NEED_EPS_ADJUST;
    *reinterpret_cast<float*>(tensor->op_params) = eps;
}

static inline float get_tensor_eps(const ggml_tensor* tensor)
{
    return (tensor->flags & GGML_TENSOR_FLAG_NEED_EPS_ADJUST)
        ? *reinterpret_cast<const float*>(tensor->op_params)
        : 0.0f;
}

void BasicModule::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_TENSOR(weight);
    LOAD_OPTIONAL_TENSOR(bias);
}

void LayerNorm::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_OPTIONAL_TENSOR(weight);
    LOAD_OPTIONAL_TENSOR(bias);
}

void CausalConvPositionEmbedding::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE_EX("conv1.0", conv1);
    LOAD_SUBMODULE_EX("conv2.0", conv2);
}

void InputEmbedding::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(proj);
    LOAD_SUBMODULE(conv_pos_embed);
}

void TimestepEmbedding::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE_EX("time_mlp.0", time_mlp_0);
    LOAD_SUBMODULE_EX("time_mlp.2", time_mlp_2);
}

void AdaLayerNormZero::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(linear);
    LOAD_SUBMODULE(norm);
}

void Attention::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(to_q);
    LOAD_SUBMODULE(to_k);
    LOAD_SUBMODULE(to_v);
    LOAD_SUBMODULE_EX("to_out.0", to_out);
}

void FeedForward::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE_EX("ff.0.0", ff_0_0);
    LOAD_SUBMODULE_EX("ff.2", ff_2);
}

void DiTBlock::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(attn_norm);
    LOAD_SUBMODULE(attn);
    LOAD_SUBMODULE(ff_norm);
    LOAD_SUBMODULE(ff);
}

void AdaLayerNorm_Final::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(linear);
    LOAD_SUBMODULE(norm);
}

void DiT::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(time_embed);
    LOAD_SUBMODULE(input_embed);

    int heads;
    int depth;
    LOAD_METADATA(heads);
    LOAD_METADATA(depth);
    LOAD_METADATA(mel_dim);

    transformer_blocks.resize(depth);
    for (int i = 0; i != depth; ++i)
    {
        auto& block = transformer_blocks[i];
        auto name = std::format("{}.transformer_blocks.{}", prefix, i);
        block.OnLoad(loader, name);
        block.attn.heads = heads;
    }

    LOAD_SUBMODULE(norm_out);
    LOAD_SUBMODULE(proj_out);
}

void CausalConditionalCFM::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(estimator);

    LOAD_METADATA(inference_cfg_rate);

    t_span.resize(diffusion_steps + 1);
    for (int i = 0; i <= diffusion_steps; ++i)
        t_span[i] = 1.f - std::cos(0.5f * 3.14159265358979323846f * i / diffusion_steps);
}

void PreLookaheadLayer::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_METADATA(pre_lookahead_len);

    LOAD_SUBMODULE(conv1);
    LOAD_SUBMODULE(conv2);
}

void CausalMaskedDiffWithDiT::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_METADATA(token_mel_ratio);

    LOAD_TENSOR_EX("input_embedding.weight", input_embedding);
    LOAD_SUBMODULE(spk_embed_affine_layer);
    LOAD_SUBMODULE(pre_lookahead_layer);
    LOAD_SUBMODULE(decoder);
}

void CausalConvRNNF0Predictor::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE_EX("condnet.0", condnet_0);
    LOAD_SUBMODULE_EX("condnet.2", condnet_2);
    LOAD_SUBMODULE_EX("condnet.4", condnet_4);
    LOAD_SUBMODULE_EX("condnet.6", condnet_6);
    LOAD_SUBMODULE_EX("condnet.8", condnet_8);
    LOAD_SUBMODULE(classifier);
}

void SourceModuleHnNSF::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(l_linear);
}

void Snake::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_TENSOR(alpha);

    set_tensor_eps(alpha, 1e-6f);
}

void ResBlock::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    int64_t id;
    GGML_ASSERT(loader.find_metadata_key(combine_prefix(prefix, "dilations").c_str(), id));
    GGML_ASSERT(loader.parser.arr_type(id) == GGUF_TYPE_INT32);

    const auto n_dilations = loader.parser.arr_n(id);
    const int* dilations = reinterpret_cast<const int*>(loader.parser.arr_data(id));
    convs.resize(n_dilations);
    for (size_t i = 0; i != n_dilations; ++i)
    {
        auto& [snk1, conv1, snk2, conv2] = convs[i];
        conv1.causal_type = CausalConv1d::causal_type_t::left;
        conv1.d = dilations[i];
        conv2.causal_type = CausalConv1d::causal_type_t::left;
        conv2.d = 1;

        LOAD_SUBMODULE_EX(std::format("convs1.{}", i).c_str(), conv1);
        LOAD_SUBMODULE_EX(std::format("convs2.{}", i).c_str(), conv2);
        LOAD_SUBMODULE_EX(std::format("activations1.{}", i).c_str(), snk1);
        LOAD_SUBMODULE_EX(std::format("activations2.{}", i).c_str(), snk2);
    }
}

void CausalHiFTGenerator::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(f0_predictor);
    LOAD_SUBMODULE(m_source);
    LOAD_SUBMODULE(conv_pre);
    conv_pre.causal_type = CausalConv1d::causal_type_t::right;

    int num_kernels;
    GGML_ASSERT(loader.get_metadata("sample_rate", reinterpret_cast<uint32_t&>(sampling_rate)));
    LOAD_METADATA(num_kernels);
    LOAD_METADATA(nb_harmonics);
    LOAD_METADATA(nsf_alpha);
    LOAD_METADATA(nsf_voiced_threshold);
    LOAD_METADATA(nsf_sigma);
    LOAD_METADATA(lrelu_slope);
    LOAD_METADATA(audio_limit);
    GGML_ASSERT(loader.get_metadata(prefix, "istft_params.n_fft", nfft));
    GGML_ASSERT(loader.get_metadata(prefix, "istft_params.hop_len", hop_len));

    int64_t id;
    GGML_ASSERT(loader.find_metadata_key(combine_prefix(prefix, "upsample_rates").c_str(), id));
    GGML_ASSERT(loader.parser.arr_type(id) == GGUF_TYPE_INT32);

    auto layers = loader.parser.arr_n(id);
    auto upsample_rates = reinterpret_cast<const int*>(loader.parser.arr_data(id));

    ups.resize(layers);
    scale_factor = hop_len;
    for (size_t i = 0; i != layers; ++i)
    {
        auto& up = ups[i];
        int upsample_rate = upsample_rates[i];
        scale_factor *= upsample_rate;
        up.s = upsample_rate;
        LOAD_SUBMODULE_EX(std::format("ups.{}", i).c_str(), up);
    }
    
    source_downs.resize(layers);
    source_resblocks.resize(layers);
    for (size_t i = 0; i != layers - 1; ++i)
    {
        auto down = new CausalConv1dDownSample;
        down->s = 1;
        for (const auto rate : std::ranges::reverse_view(std::span(upsample_rates + 1 + i, layers - i - 1)))
            down->s *= rate;
        source_downs[i].reset(down);

        LOAD_SUBMODULE_EX(std::format("source_downs.{}", i).c_str(), *down);
        LOAD_SUBMODULE_EX(std::format("source_resblocks.{}", i).c_str(), source_resblocks[i]);
    }
    // Final block.
    {
        const auto layer_idx = layers - 1;
        auto conv = new CausalConv1d;
        conv->d = 1;
        conv->causal_type = CausalConv1d::causal_type_t::left;
        source_downs[layer_idx].reset(conv);

        LOAD_SUBMODULE_EX(std::format("source_downs.{}", layer_idx).c_str(), *conv);
        LOAD_SUBMODULE_EX(std::format("source_resblocks.{}", layer_idx).c_str(), source_resblocks[layer_idx]);
    }

    layers *= num_kernels;
    resblocks.resize(layers);
    for (size_t i = 0; i != layers; ++i)
        LOAD_SUBMODULE_EX(std::format("resblocks.{}", i).c_str(), resblocks[i]);

    LOAD_SUBMODULE(conv_post);
    conv_post.causal_type = CausalConv1d::causal_type_t::left;
}

void Qwen2MLP::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(gate_proj);
    LOAD_SUBMODULE(up_proj);
    LOAD_SUBMODULE(down_proj);
}

void Qwen2RMSNorm::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_TENSOR(weight);
}

void Qwen2Attention::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(q_proj);
    LOAD_SUBMODULE(k_proj);
    LOAD_SUBMODULE(v_proj);
    LOAD_SUBMODULE(o_proj);
}

void Qwen2DecoderLayer::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(self_attn);
    LOAD_SUBMODULE(mlp);
    LOAD_SUBMODULE(input_layernorm);
    LOAD_SUBMODULE(post_attention_layernorm);
}

void CosyVoice3LM::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_TENSOR_EX("embed_tokens.weight", embed_tokens_weight);
    LOAD_TENSOR_EX("speech_embedding.weight", speech_embedding_weight);
    LOAD_SUBMODULE(norm);
    LOAD_SUBMODULE(llm_decoder);

    int num_hidden_layers;
    LOAD_METADATA(num_hidden_layers);
    LOAD_METADATA(num_attention_heads);
    LOAD_METADATA(num_key_value_heads);
    LOAD_METADATA(rms_norm_eps);
    LOAD_METADATA(rope_theta);
    LOAD_METADATA(sos_token_id);
    LOAD_METADATA(task_token_id);

    GGML_ASSERT(embed_tokens_weight->ne[0] == speech_embedding_weight->ne[0]);
    layers.resize(num_hidden_layers);

    auto layers_prefix = combine_prefix(prefix, "layers");
    for (int i = 0; i != num_hidden_layers; ++i)
    {
        auto& layer = layers[i];
        auto name = std::format("{}.{}", layers_prefix, i);
        LOAD_SUBMODULE_EX(name.c_str(), layer);
    }
}

static ggml_type cosyvoice_kv_cache_type_to_ggml(cosyvoice_kv_cache_type_t t)
{
    switch (t)
    {
    case COSYVOICE_KV_CACHE_TYPE_F32: return GGML_TYPE_F32;
    case COSYVOICE_KV_CACHE_TYPE_F16: return GGML_TYPE_F16;
    case COSYVOICE_KV_CACHE_TYPE_Q8_0: return GGML_TYPE_Q8_0;
    case COSYVOICE_KV_CACHE_TYPE_Q5_1: return GGML_TYPE_Q5_1;
    case COSYVOICE_KV_CACHE_TYPE_Q5_0: return GGML_TYPE_Q5_0;
    case COSYVOICE_KV_CACHE_TYPE_Q4_1: return GGML_TYPE_Q4_1;
    case COSYVOICE_KV_CACHE_TYPE_Q4_0: return GGML_TYPE_Q4_0;
    default: GGML_ABORT("unexpected kv cache type");
    }
}

static cosyvoice_kv_cache_type_t cosyvoice_ggml_to_kv_cache_type(ggml_type t)
{
    switch (t)
    {
    case GGML_TYPE_F32: return COSYVOICE_KV_CACHE_TYPE_F32;
    case GGML_TYPE_F16: return COSYVOICE_KV_CACHE_TYPE_F16;
    case GGML_TYPE_Q8_0: return COSYVOICE_KV_CACHE_TYPE_Q8_0;
    case GGML_TYPE_Q5_1: return COSYVOICE_KV_CACHE_TYPE_Q5_1;
    case GGML_TYPE_Q5_0: return COSYVOICE_KV_CACHE_TYPE_Q5_0;
    case GGML_TYPE_Q4_1: return COSYVOICE_KV_CACHE_TYPE_Q4_1;
    case GGML_TYPE_Q4_0: return COSYVOICE_KV_CACHE_TYPE_Q4_0;
    default: GGML_ABORT("unexpected ggml type for kv cache");
    }
}

static ggml_type cosyvoice_get_kv_fallback_type(ggml_type t)
{
    switch (t)
    {
    case GGML_TYPE_Q4_0: return GGML_TYPE_Q4_1;
    case GGML_TYPE_Q4_1: return GGML_TYPE_Q5_0;
    case GGML_TYPE_Q5_0: return GGML_TYPE_Q5_1;
    case GGML_TYPE_Q5_1: return GGML_TYPE_Q8_0;
    case GGML_TYPE_Q8_0: return GGML_TYPE_F16;
    case GGML_TYPE_F16:  return GGML_TYPE_F32;
    default: GGML_ABORT("fatal error");
    }
}

union kv_cache_type_union
{
    struct
    {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__) || defined(_BYTE_ORDER) && (_BYTE_ORDER == _BIG_ENDIAN) || defined(__BIG_ENDIAN__) && !defined(__LITTLE_ENDIAN__) || defined(__ARMEB__) || defined(__MIPSEB__) || defined(__sparc__)
        uint32_t                    kv_cache_separate_buffers : 1;
        cosyvoice_kv_cache_type_t : 16;
        cosyvoice_kv_cache_type_t   kv_cache_fallback : 5;
        cosyvoice_kv_cache_type_t   v_cache_type : 5;
        cosyvoice_kv_cache_type_t   k_cache_type : 5;
#else
        cosyvoice_kv_cache_type_t   k_cache_type : 5;
        cosyvoice_kv_cache_type_t   v_cache_type : 5;
        cosyvoice_kv_cache_type_t   kv_cache_fallback : 5;
        cosyvoice_kv_cache_type_t : 16;
        uint32_t                    kv_cache_separate_buffers : 1;
#endif
    };
    cosyvoice_kv_cache_type_t       kv_cache_type;
};

static std::tuple<ggml_type, ggml_type> cosyvoice_check_kv_cache_types(
    ggml_context* ctx, ggml_backend_t backend,
    bool& fattn, int num_attn_heads, int num_kv_heads,
    const Linear& q_proj, const Linear& k_proj, const Linear& v_proj,
    bool kv_fallback, kv_cache_type_union& kv_type_union)
{
    ggml_type k_type, v_type;

    if (fattn)
    {
        auto fattn_check = [&](ggml_type check_k, ggml_type check_v) -> bool
        {
            const auto k_head_dim = k_proj.weight->ne[1] / num_kv_heads;
            const auto v_head_dim = v_proj.weight->ne[1] / num_kv_heads;

            auto position_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            auto q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, q_proj.weight->ne[1] / num_attn_heads, 1, num_attn_heads, 1);
            auto k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k_head_dim * num_kv_heads, 1, 1);
            auto v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, v_head_dim * num_kv_heads, 1, 1);
            auto cached_k = ggml_new_tensor_3d(ctx, check_k, k_head_dim * num_kv_heads, 1, 1);
            auto cached_v = ggml_new_tensor_3d(ctx, check_v, v_head_dim * num_kv_heads, 1, 1);

            cached_k = ggml_set_rows(ctx, cached_k, k, position_ids);
            cached_v = ggml_set_rows(ctx, cached_v, v, position_ids);
            if (!ggml_backend_supports_op(backend, cached_k) || !ggml_backend_supports_op(backend, cached_v))
                return false;

            cached_k = ggml_view_4d(ctx, cached_k, k_head_dim, num_kv_heads, 1, 1, cached_k->nb[1] / num_kv_heads, cached_k->nb[1], cached_k->nb[2], 0);
            cached_k = ggml_permute(ctx, cached_k, 0, 2, 1, 3);
            cached_v = ggml_view_4d(ctx, cached_v, v_head_dim, num_kv_heads, 1, 1, cached_v->nb[1] / num_kv_heads, cached_v->nb[1], cached_v->nb[2], 0);
            cached_v = ggml_permute(ctx, cached_v, 0, 2, 1, 3);

            auto o = ggml_flash_attn_ext(ctx, q, cached_k, cached_v, nullptr, 1.f / std::sqrt(static_cast<float>(k_head_dim)), 0.f, 0.f);
            return ggml_backend_supports_op(backend, o);
        };

        fattn = fattn_check(GGML_TYPE_F32, GGML_TYPE_F32);
        if (fattn)
        {
            if (kv_fallback)
            {
                ggml_type cur_type;

                do
                {
                    if (kv_type_union.kv_cache_separate_buffers)
                    {
                        if (k_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.k_cache_type),
                            v_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.v_cache_type);
                            fattn_check(k_type, v_type))
                        {
                            kv_type_union.kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                                kv_type_union.k_cache_type,
                                kv_type_union.v_cache_type,
                                kv_type_union.v_cache_type);
                            break;
                        }

                        cur_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.kv_cache_fallback);
                    }
                    else
                        cur_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.kv_cache_type);

                    do
                    {
                        if (fattn_check(cur_type, cur_type))
                        {
                            k_type = cur_type;
                            v_type = cur_type;
                            kv_type_union.kv_cache_type = cosyvoice_ggml_to_kv_cache_type(cur_type);
                            break;
                        }

                        cur_type = cosyvoice_get_kv_fallback_type(cur_type);
                    } while (cur_type != GGML_TYPE_F32);

                    if (kv_type_union.kv_cache_separate_buffers)
                        kv_type_union.kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                            cosyvoice_ggml_to_kv_cache_type(k_type),
                            cosyvoice_ggml_to_kv_cache_type(v_type),
                            cosyvoice_ggml_to_kv_cache_type(v_type));
                } while (false);
            }
            else if (kv_type_union.kv_cache_separate_buffers)
            {
                k_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.k_cache_type);
                v_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.v_cache_type);
                fattn = fattn_check(k_type, v_type);
                if (fattn)
                    kv_type_union.kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                        kv_type_union.k_cache_type,
                        kv_type_union.v_cache_type,
                        kv_type_union.v_cache_type);
            }
            else
            {
                k_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.kv_cache_type);
                v_type = k_type;
                fattn = fattn_check(k_type, v_type);
            }
        }
    }

    if (!fattn)
    {
        ggml_type cur_type;
        auto attn_check = [&](ggml_type check_k, ggml_type check_v) -> bool
        {
            if (ggml_is_quantized(check_v)) return false;

            const auto k_head_dim = k_proj.weight->ne[1] / num_kv_heads;
            const auto v_head_dim = v_proj.weight->ne[1] / num_kv_heads;

            auto position_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            auto q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, q_proj.weight->ne[1] / num_attn_heads, 1, num_attn_heads, 1);
            auto k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k_head_dim * num_kv_heads, 1, 1);
            auto cached_k = ggml_new_tensor_3d(ctx, check_k, k_head_dim * num_kv_heads, 1, 1);

            // K: head-merged cache rows scattered by position ids, then viewed back into
            // heads and permuted (see cosyvoice_kv_cache::update_cache)
            cached_k = ggml_set_rows(ctx, cached_k, k, position_ids);
            cached_k = ggml_view_4d(ctx, cached_k, k_head_dim, num_kv_heads, 1, 1, cached_k->nb[1] / num_kv_heads, cached_k->nb[1], cached_k->nb[2], 0);
            cached_k = ggml_permute(ctx, cached_k, 0, 2, 1, 3);

            // V: stored transposed and updated element-wise through the flattened cache
            // via v_idxs (see cosyvoice_kv_cache::update_cache)
            auto v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, v_head_dim * num_kv_heads);
            auto cached_v = ggml_new_tensor_2d(ctx, check_v, 1, v_head_dim * num_kv_heads);
            auto v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, v_head_dim * num_kv_heads);
            auto v_flat = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));
            auto cached_v_flat = ggml_reshape_2d(ctx, cached_v, 1, ggml_nelements(cached_v));
            cached_v = ggml_set_rows(ctx, cached_v_flat, v_flat, v_idxs);
            cached_v = ggml_view_4d(ctx, cached_v, 1, v_head_dim, num_kv_heads, 1, cached_v->nb[1], v_head_dim * cached_v->nb[1], cached_v->nb[2], 0);

            if (!ggml_backend_supports_op(backend, cached_k) || !ggml_backend_supports_op(backend, cached_v))
                return false;

            auto s = ggml_mul_mat(ctx, cached_k, q);
            auto o = ggml_mul_mat(ctx, cached_v, s);
            return ggml_backend_supports_op(backend, o) && ggml_backend_supports_op(backend, s);
        };

        do
        {
            if (kv_type_union.kv_cache_separate_buffers)
            {
                if (k_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.k_cache_type),
                    v_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.v_cache_type);
                    attn_check(k_type, v_type))
                {
                    kv_type_union.kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                        kv_type_union.k_cache_type,
                        kv_type_union.v_cache_type,
                        kv_type_union.v_cache_type);
                    break;
                }
                else if (kv_fallback && attn_check(k_type, GGML_TYPE_F16))
                {
                    k_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.k_cache_type);
                    v_type = GGML_TYPE_F16;
                    kv_type_union.kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                        kv_type_union.k_cache_type,
                        COSYVOICE_KV_CACHE_TYPE_F16,
                        COSYVOICE_KV_CACHE_TYPE_F16);
                    break;
                }
                else
                    cur_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.kv_cache_fallback);
            }
            else
                cur_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.kv_cache_type);

            if (kv_type_union.kv_cache_separate_buffers)
            {
                v_type = ggml_is_quantized(cur_type) ? GGML_TYPE_F16 : cur_type;
                do
                {
                    if (attn_check(cur_type, v_type))
                    {
                        k_type = cur_type;
                        break;
                    }

                    GGML_ASSERT(kv_fallback);
                    cur_type = cosyvoice_get_kv_fallback_type(cur_type);
                } while (cur_type != GGML_TYPE_F32);

                kv_type_union.kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                    cosyvoice_ggml_to_kv_cache_type(cur_type),
                    cosyvoice_ggml_to_kv_cache_type(v_type),
                    cosyvoice_ggml_to_kv_cache_type(v_type));

                k_type = cur_type;
            }
            else
            {
                if (attn_check(cur_type, cur_type))
                {
                    k_type = cur_type;
                    v_type = cur_type;
                    kv_type_union.kv_cache_type = cosyvoice_ggml_to_kv_cache_type(cur_type);
                    break;
                }
                else
                {
                    GGML_ASSERT(kv_fallback);
                    cur_type = GGML_TYPE_F16;
                }
                k_type = cur_type;
                v_type = cur_type;
                kv_type_union.kv_cache_type = cosyvoice_ggml_to_kv_cache_type(cur_type);
            }
        } while (false);
    }

    return { k_type, v_type };
}

void cosyvoice_model_3::load(gguf_loader& loader)
{
    auto& flow = cv3_shared->flow;
    auto& hift = cv3_shared->hift;
    auto& llm = cv3_shared->llm;

    {
        // Resolve the effective number of diffusion steps before the flow submodule
        // is loaded (which sizes t_span) and before the DiT KV slots are clamped.
        // Precedence: params override (v4) > GGUF metadata > default 10, clamped to [1, MAX].
        int32_t md_steps = 10;
        loader.get_metadata("decoder", "diffusion_steps", md_steps);

        const int32_t req_steps = shared->params.diffusion_steps;
        int diffusion_steps = req_steps > 0 ? req_steps : md_steps;
        if (diffusion_steps < 1)
            diffusion_steps = 1;
        if (diffusion_steps > CausalConditionalCFM::MAX_DIFFUSION_STEPS)
        {
            cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_WARN,
                std::format("decoder.diffusion_steps {} clamped to the maximum of {}.\n", diffusion_steps, CausalConditionalCFM::MAX_DIFFUSION_STEPS).c_str());
            diffusion_steps = CausalConditionalCFM::MAX_DIFFUSION_STEPS;
        }
        flow.decoder.diffusion_steps = diffusion_steps;

        auto& n_fixed_slots = shared->params.dit_kv_fixed_slots;
        auto& n_offloadable_slots = shared->params.dit_kv_offloadable_slots;
        if (n_fixed_slots > diffusion_steps)
            n_fixed_slots = diffusion_steps;
        if (n_offloadable_slots + n_fixed_slots > diffusion_steps)
            n_offloadable_slots = diffusion_steps - n_fixed_slots;

        // Physical slot counts backing the logical slots. 0 disables sharing (one
        // physical slot per logical slot); a positive value merges adjacent steps
        // into that many groups, each sharing one KV cache.
        auto& actual_fixed = shared->params.dit_kv_actual_fixed_slots;
        auto& actual_offloadable = shared->params.dit_kv_actual_offloadable_slots;
        if (actual_fixed == 0 || actual_fixed > n_fixed_slots)
            actual_fixed = n_fixed_slots;
        if (actual_offloadable == 0 || actual_offloadable > n_offloadable_slots)
            actual_offloadable = n_offloadable_slots;
    }


    flow.OnLoad(loader, {});
    hift.OnLoad(loader, {});
    llm.OnLoad(loader, {});

    auto& tensors = loader.tensors;

    ggml_init_params params =
    {
        .mem_size = (tensors.size() + 512 + 4 * shared->params.n_workers) * ggml_tensor_overhead(),
        .mem_buffer = nullptr,
        .no_alloc = true
    };

    // init sinusodal position embedding
    constexpr int dim = 256;
    constexpr int half_dim = dim / 2;

    auto emb_buffer = std::make_unique<float[]>(half_dim);
    const auto emb = std::log(10000.f) / (half_dim - 1);

    for (int i = 0; i != half_dim; ++i)
        emb_buffer[i] = std::exp(i * -emb) * 1000;

    auto backend = worker->backend.get();
    auto cpu_backend = worker->cpu_backend.get();
    auto buft = ggml_backend_get_default_buffer_type(backend);
    auto alignment = ggml_backend_buft_get_alignment(buft);

    // Precompute all DiT AdaLN modulation vectors on the active inference backend for the
    // fixed diffusion schedule t_span[0..diffusion_steps-1]. Evaluating through the exact
    // same backend ops (TimestepEmbedding + 22x AdaLayerNormZero.linear + AdaLayerNorm_Final.linear
    // + scale_bias(+1)) guarantees bit-exact parity with on-the-fly execution on every backend
    // while freeing ~283 MiB of F16 weights from permanent VRAM (~5.5 MiB F32 table retained).
    std::unordered_set<ggml_tensor**> precomputed_adaln_tensors;
    std::vector<float> adaln_host;
    int64_t adaln_dim = 0;
    int64_t adaln_rows = 0;
    const int adaln_steps = flow.decoder.diffusion_steps;
    if (adaln_steps > 0)
    {
        auto& dit = flow.decoder.estimator;
        const size_t depth = dit.transformer_blocks.size();
        auto linear_eligible = [](const Linear& lin)
        {
            if (!lin.weight || lin.lora_a || lin.lora_b || lin.lora_enabled)
                return false;
            if (get_tensor_eps(lin.weight) != 0.f)
                return false;
            if (lin.bias && get_tensor_eps(lin.bias) != 0.f)
                return false;
            return true;
        };
        bool can_precompute = depth > 0
            && linear_eligible(dit.time_embed.time_mlp_0)
            && linear_eligible(dit.time_embed.time_mlp_2)
            && linear_eligible(dit.norm_out.linear)
            && dit.norm_out.linear.weight->ne[1] > 0
            && (dit.norm_out.linear.weight->ne[1] % 2) == 0;
        if (can_precompute)
        {
            adaln_dim = dit.norm_out.linear.weight->ne[1] / 2;
            for (size_t b = 0; b != depth; ++b)
            {
                const auto& lin = dit.transformer_blocks[b].attn_norm.linear;
                if (!linear_eligible(lin) || lin.weight->ne[1] != 6 * adaln_dim)
                {
                    can_precompute = false;
                    break;
                }
            }
        }
        if (can_precompute)
        {
            auto aligned_alloc_size = [&](const ggml_tensor* t) -> size_t
            {
                return t ? get_aligned_size(ggml_backend_buft_get_alloc_size(buft, t), alignment) : 0;
            };
            const size_t emb_bytes = get_aligned_size(sizeof(float) * half_dim, alignment);
            const size_t t_pair_bytes = get_aligned_size(sizeof(float) * 2, alignment);
            size_t tmp_w_bytes = emb_bytes
                + aligned_alloc_size(dit.time_embed.time_mlp_0.weight)
                + aligned_alloc_size(dit.time_embed.time_mlp_0.bias)
                + aligned_alloc_size(dit.time_embed.time_mlp_2.weight)
                + aligned_alloc_size(dit.time_embed.time_mlp_2.bias)
                + t_pair_bytes * static_cast<size_t>(adaln_steps);
            for (size_t b = 0; b != depth; ++b)
            {
                const auto& lin = dit.transformer_blocks[b].attn_norm.linear;
                tmp_w_bytes = std::max(tmp_w_bytes, aligned_alloc_size(lin.weight) + aligned_alloc_size(lin.bias));
            }
            tmp_w_bytes = std::max(tmp_w_bytes,
                aligned_alloc_size(dit.norm_out.linear.weight) + aligned_alloc_size(dit.norm_out.linear.bias));

            const int64_t mlp2_out = dit.time_embed.time_mlp_2.weight->ne[1];
            const size_t temb_slot_bytes = get_aligned_size(sizeof(float) * static_cast<size_t>(mlp2_out * 2), alignment);
            const size_t temb_total_bytes = temb_slot_bytes * static_cast<size_t>(adaln_steps);
            const size_t backend_max_buf = ggml_backend_get_max_size(backend);

            if (tmp_w_bytes <= backend_max_buf && temb_total_bytes <= backend_max_buf)
            {
                ggml_backend_buffer_ptr tmp_w_buf(ggml_backend_buft_alloc_buffer(buft, tmp_w_bytes));
                ggml_backend_buffer_ptr tmp_temb_buf(ggml_backend_buft_alloc_buffer(buft, temb_total_bytes));
                const size_t graph_size = static_cast<size_t>(std::max(2048, adaln_steps * 32));
                ggml_backend_t sched_backends[] = { backend, cpu_backend };
                ggml_backend_sched_ptr tmp_sched(ggml_backend_sched_new(
                    sched_backends,
                    nullptr,
                    backend == cpu_backend ? 1 : 2,
                    graph_size,
                    false,
                    true));

                ggml_context_ptr temb_ctx(ggml_init({
                    .mem_size = static_cast<size_t>(adaln_steps + 4) * ggml_tensor_overhead(),
                    .mem_buffer = nullptr,
                    .no_alloc = true
                }));
                ggml_context_ptr graph_ctx(ggml_init({
                    .mem_size = ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
                    .mem_buffer = nullptr,
                    .no_alloc = true
                }));

                auto assign_backends = [&](ggml_cgraph* gf)
                {
                    const auto& op_caps = shared->op_caps;
                    auto is_virtual = [](ggml_op op)
                    {
                        return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
                    };
                    for (auto* node : ggml_cgraph_node_iterator(gf))
                    {
                        if (node->data)
                            continue;
                        auto op = node->op;
                        auto* src = node;
                        while (src && is_virtual(src->op))
                            src = src->src[0];
                        if (src)
                            op = src->op;
                        ggml_backend_t target = backend;
                        if (op == GGML_OP_SIN && !op_caps.sin)
                            target = cpu_backend;
                        else if (op == GGML_OP_COS && !op_caps.cos)
                            target = cpu_backend;
                        ggml_backend_sched_set_tensor_backend(tmp_sched.get(), node, target);
                    }
                };

                bool ok = tmp_w_buf && tmp_temb_buf && tmp_sched && temb_ctx && graph_ctx;
                std::vector<ggml_tensor*> t_emb_s(static_cast<size_t>(adaln_steps), nullptr);

                // Stage 1: Compute TimestepEmbedding(t_span[s]) -> silu for all steps on backend.
                if (ok)
                {
                    ggml_context_ptr stage1_ctx(ggml_init({
                        .mem_size = static_cast<size_t>(adaln_steps + 16) * ggml_tensor_overhead(),
                        .mem_buffer = nullptr,
                        .no_alloc = true
                    }));
                    if (!stage1_ctx)
                        ok = false;
                    else
                    {
                    char* w_cur = reinterpret_cast<char*>(ggml_backend_buffer_get_base(tmp_w_buf.get()));
                    auto stage_copy = [&](const ggml_tensor* orig) -> ggml_tensor*
                    {
                        if (!orig) return nullptr;
                        ggml_tensor* t = ggml_new_tensor(stage1_ctx.get(), orig->type, GGML_MAX_DIMS, orig->ne);
                        ggml_backend_tensor_alloc(tmp_w_buf.get(), t, w_cur);
                        w_cur += aligned_alloc_size(t);
                        ggml_backend_tensor_set(t, ggml_get_data(orig), 0, ggml_nbytes(orig));
                        return t;
                    };

                    TimestepEmbedding tmp_te;
                    tmp_te.time_embed.emb = ggml_new_tensor_1d(stage1_ctx.get(), GGML_TYPE_F32, half_dim);
                    ggml_backend_tensor_alloc(tmp_w_buf.get(), tmp_te.time_embed.emb, w_cur);
                    w_cur += emb_bytes;
                    ggml_backend_tensor_set(tmp_te.time_embed.emb, emb_buffer.get(), 0, sizeof(float) * half_dim);
                    tmp_te.time_mlp_0.weight = stage_copy(dit.time_embed.time_mlp_0.weight);
                    tmp_te.time_mlp_0.bias   = stage_copy(dit.time_embed.time_mlp_0.bias);
                    tmp_te.time_mlp_2.weight = stage_copy(dit.time_embed.time_mlp_2.weight);
                    tmp_te.time_mlp_2.bias   = stage_copy(dit.time_embed.time_mlp_2.bias);

                    ggml_cgraph* gf = ggml_new_graph_custom(graph_ctx.get(), graph_size, false);
                    std::vector<ggml_tensor*> te_outs(static_cast<size_t>(adaln_steps), nullptr);
                    for (int s = 0; s != adaln_steps; ++s)
                    {
                        const float t_val = flow.decoder.get_t_and_dt(nullptr, s + 1)[0];
                        const float t_pair[2] = { t_val, t_val };
                        ggml_tensor* t_in = ggml_new_tensor_1d(stage1_ctx.get(), GGML_TYPE_F32, 2);
                        ggml_backend_tensor_alloc(tmp_w_buf.get(), t_in, w_cur);
                        w_cur += t_pair_bytes;
                        ggml_backend_tensor_set(t_in, t_pair, 0, sizeof(t_pair));

                        ggml_tensor* te = tmp_te.build_cgraph(graph_ctx.get(), t_in);
                        te = ggml_silu(graph_ctx.get(), te);
                        ggml_set_output(te);
                        ggml_build_forward_expand(gf, te);
                        te_outs[static_cast<size_t>(s)] = te;
                    }
                    assign_backends(gf);
                    if (!ggml_backend_sched_alloc_graph(tmp_sched.get(), gf)
                        || ggml_backend_sched_graph_compute(tmp_sched.get(), gf) != GGML_STATUS_SUCCESS)
                    {
                        ok = false;
                    }
                    else
                    {
                        char* te_cur = reinterpret_cast<char*>(ggml_backend_buffer_get_base(tmp_temb_buf.get()));
                        std::vector<float> te_host(static_cast<size_t>(mlp2_out * 2));
                        const size_t te_nbytes = te_host.size() * sizeof(float);
                        for (int s = 0; s != adaln_steps; ++s)
                        {
                            ggml_backend_tensor_get(te_outs[static_cast<size_t>(s)], te_host.data(), 0, te_nbytes);
                            ggml_tensor* ts = ggml_new_tensor_2d(temb_ctx.get(), GGML_TYPE_F32, mlp2_out, 2);
                            ggml_backend_tensor_alloc(tmp_temb_buf.get(), ts, te_cur);
                            te_cur += temb_slot_bytes;
                            ggml_backend_tensor_set(ts, te_host.data(), 0, te_nbytes);
                            t_emb_s[static_cast<size_t>(s)] = ts;
                        }
                    }
                    }
                }

                // Stage 2: Stream each block's AdaLN linear + final norm linear through tmp_w_buf.
                if (ok)
                {
                    adaln_rows = static_cast<int64_t>(depth) * 6 + 2;
                    adaln_host.assign(static_cast<size_t>(adaln_dim * adaln_rows * adaln_steps), 0.0f);
                    const size_t step_stride = static_cast<size_t>(adaln_dim * adaln_rows);

                    ggml_context_ptr lin_ctx(ggml_init({
                        .mem_size = 16 * ggml_tensor_overhead(),
                        .mem_buffer = nullptr,
                        .no_alloc = true
                    }));
                    if (!lin_ctx)
                        ok = false;

                    auto eval_staged_linear = [&](const Linear& src_lin, int64_t n_rows, int64_t row_base,
                                                  std::initializer_list<int64_t> scale_row_offsets) -> bool
                    {
                        ggml_reset(lin_ctx.get());
                        ggml_reset(graph_ctx.get());
                        ggml_backend_sched_reset(tmp_sched.get());

                        char* w_cur = reinterpret_cast<char*>(ggml_backend_buffer_get_base(tmp_w_buf.get()));
                        Linear tmp_lin;
                        tmp_lin.weight = ggml_new_tensor(lin_ctx.get(), src_lin.weight->type, GGML_MAX_DIMS, src_lin.weight->ne);
                        ggml_backend_tensor_alloc(tmp_w_buf.get(), tmp_lin.weight, w_cur);
                        w_cur += aligned_alloc_size(tmp_lin.weight);
                        ggml_backend_tensor_set(tmp_lin.weight, ggml_get_data(src_lin.weight), 0, ggml_nbytes(src_lin.weight));
                        if (src_lin.bias)
                        {
                            tmp_lin.bias = ggml_new_tensor(lin_ctx.get(), src_lin.bias->type, GGML_MAX_DIMS, src_lin.bias->ne);
                            ggml_backend_tensor_alloc(tmp_w_buf.get(), tmp_lin.bias, w_cur);
                            w_cur += aligned_alloc_size(tmp_lin.bias);
                            ggml_backend_tensor_set(tmp_lin.bias, ggml_get_data(src_lin.bias), 0, ggml_nbytes(src_lin.bias));
                        }

                        ggml_cgraph* gf = ggml_new_graph_custom(graph_ctx.get(), graph_size, false);
                        std::vector<ggml_tensor*> emb_outs(static_cast<size_t>(adaln_steps), nullptr);
                        std::vector<std::vector<ggml_tensor*>> scale_outs(
                            static_cast<size_t>(adaln_steps), std::vector<ggml_tensor*>(scale_row_offsets.size(), nullptr));
                        for (int s = 0; s != adaln_steps; ++s)
                        {
                            ggml_tensor* emb_s = tmp_lin.build_cgraph(graph_ctx.get(), t_emb_s[static_cast<size_t>(s)]);
                            ggml_set_output(emb_s);
                            ggml_build_forward_expand(gf, emb_s);
                            emb_outs[static_cast<size_t>(s)] = emb_s;

                            size_t k = 0;
                            for (int64_t r_off : scale_row_offsets)
                            {
                                ggml_tensor* sv = ggml_view_1d(graph_ctx.get(), emb_s, adaln_dim,
                                    static_cast<size_t>(r_off * adaln_dim) * sizeof(float));
                                ggml_tensor* sp1 = ggml_scale_bias(graph_ctx.get(), ggml_cont(graph_ctx.get(), sv), 1.f, 1.f);
                                ggml_set_output(sp1);
                                ggml_build_forward_expand(gf, sp1);
                                scale_outs[static_cast<size_t>(s)][k++] = sp1;
                            }
                        }
                        assign_backends(gf);
                        if (!ggml_backend_sched_alloc_graph(tmp_sched.get(), gf)
                            || ggml_backend_sched_graph_compute(tmp_sched.get(), gf) != GGML_STATUS_SUCCESS)
                            return false;

                        for (int s = 0; s != adaln_steps; ++s)
                        {
                            float* dst = adaln_host.data()
                                + static_cast<size_t>(s) * step_stride
                                + static_cast<size_t>(row_base * adaln_dim);
                            ggml_backend_tensor_get(emb_outs[static_cast<size_t>(s)], dst, 0,
                                static_cast<size_t>(n_rows * adaln_dim) * sizeof(float));
                            size_t k = 0;
                            for (int64_t r_off : scale_row_offsets)
                            {
                                ggml_backend_tensor_get(scale_outs[static_cast<size_t>(s)][k++],
                                    dst + static_cast<size_t>(r_off * adaln_dim), 0,
                                    static_cast<size_t>(adaln_dim) * sizeof(float));
                            }
                        }
                        return true;
                    };

                    for (size_t b = 0; b != depth && ok; ++b)
                        ok = eval_staged_linear(dit.transformer_blocks[b].attn_norm.linear, 6, static_cast<int64_t>(b * 6), { 1, 4 });
                    if (ok)
                        ok = eval_staged_linear(dit.norm_out.linear, 2, static_cast<int64_t>(depth * 6), { 0 });
                }

                ggml_backend_synchronize(backend);

                if (ok)
                {
                    auto mark_skip_linear = [&](Linear& lin)
                    {
                        if (lin.weight) { precomputed_adaln_tensors.insert(&lin.weight); lin.weight = nullptr; }
                        if (lin.bias)   { precomputed_adaln_tensors.insert(&lin.bias);   lin.bias = nullptr; }
                    };
                    mark_skip_linear(dit.time_embed.time_mlp_0);
                    mark_skip_linear(dit.time_embed.time_mlp_2);
                    for (auto& blk : dit.transformer_blocks)
                        mark_skip_linear(blk.attn_norm.linear);
                    mark_skip_linear(dit.norm_out.linear);
                }
                else
                {
                    adaln_host.clear();
                    adaln_rows = 0;
                }
            }
        }
    }

    const int mel_dim = flow.decoder.estimator.mel_dim;
    size_t mem_size = (adaln_host.empty() ? get_aligned_size(sizeof(float) * half_dim, alignment)
                                          : get_aligned_size(adaln_host.size() * sizeof(float), alignment))
        + get_aligned_size(sizeof(float) * (hift.nfft / 2 + 1), alignment)
        + get_aligned_size(sizeof(float) * (hift.nfft / 2 + 1) * hift.nfft, alignment) * 2
        + get_aligned_size(sizeof(int) * shared->params.n_batch, alignment) * shared->params.n_workers
        + get_aligned_size(sizeof(ggml_fp16_t) * shared->params.n_max_seq, alignment) * shared->params.n_workers
        + get_aligned_size(sizeof(float) * (hift.nb_harmonics + 1), alignment);
    for (const auto& [name, tensor] : tensors)
        if (tensor != &llm.embed_tokens_weight
            && tensor != &llm.speech_embedding_weight
            && !precomputed_adaln_tensors.count(tensor))
            mem_size += get_aligned_size(ggml_backend_buft_get_alloc_size(buft, *tensor), alignment);

    {
        auto add_snake_mem = [&](const std::vector<ResBlock>& blocks)
        {
            for (const auto& rb : blocks)
                for (const auto& [act1, conv1, act2, conv2] : rb.convs)
                {
                    if (act1.alpha) mem_size += get_aligned_size(ggml_backend_buft_get_alloc_size(buft, act1.alpha), alignment);
                    if (act2.alpha) mem_size += get_aligned_size(ggml_backend_buft_get_alloc_size(buft, act2.alpha), alignment);
                }
        };
        add_snake_mem(hift.resblocks);
        add_snake_mem(hift.source_resblocks);
    }

    // Split the packed weights over as many buffers as needed: some backends
    // cannot create a single allocation above a device limit (reported through
    // get_max_size) and the packed set exceeds that limit on some drivers.
    const size_t device_total = mem_size;
    size_t max_buffer_size = ggml_backend_get_max_size(backend);

    // Debug knob: override the largest weight allocation so the multi-buffer
    // path can be exercised on backends that report no limit (CUDA and CPU
    // return SIZE_MAX). The value is in bytes; 0 removes the cap.
    if (const char* env = std::getenv("COSYVOICE_MAX_WEIGHT_BUFFER_SIZE"); env && *env)
    {
        const auto* const end = env + std::strlen(env);
        size_t requested = 0;
        const auto [ptr, ec] = std::from_chars(env, end, requested);
        if (ec == std::errc{} && ptr == end)
        {
            max_buffer_size = requested != 0 ? requested : std::numeric_limits<size_t>::max();
            cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_INFO, std::format(
                "weights: max buffer size overridden to {} bytes (COSYVOICE_MAX_WEIGHT_BUFFER_SIZE)\n",
                max_buffer_size).c_str());

            constexpr size_t small_buffer_warn_bytes = 64ull * 1024ull * 1024ull;
            if (requested != 0 && requested < small_buffer_warn_bytes)
                cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_WARN, std::format(
                    "weights: COSYVOICE_MAX_WEIGHT_BUFFER_SIZE={} is very small; this creates many device "
                    "buffers and can slow down loading or fragment device memory.\n", requested).c_str());
        }
        else
            cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_WARN, std::format(
                "weights: ignoring invalid COSYVOICE_MAX_WEIGHT_BUFFER_SIZE=\"{}\"\n", env).c_str());
    }

    auto& device_buffers = shared->buffers;
    device_buffers.clear();
    size_t device_bytes_left = device_total;
    size_t device_buffer_used = 0;
    size_t device_buffer_capacity = 0;
    char* buffer_base = nullptr;

    auto alloc_device_bytes = [&](size_t size) -> char*
    {
        if (device_buffers.empty() || device_buffer_used + size > device_buffer_capacity)
        {
            if (device_buffer_capacity < device_bytes_left)
                device_bytes_left -= device_buffer_capacity;
            else
                device_bytes_left = 0;

            const auto capacity = std::max(std::min(max_buffer_size, device_bytes_left), size);
            auto buffer = ggml_backend_buft_alloc_buffer(buft, capacity);
            if (!buffer)
                throw std::runtime_error(std::format(
                    "Failed to allocate a {} byte buffer for the model weights (max {} bytes per buffer).\n",
                    capacity, max_buffer_size));

            device_buffers.emplace_back(buffer);
            buffer_base = reinterpret_cast<char*>(ggml_backend_buffer_get_base(device_buffers.back().get()));
            device_buffer_used = 0;
            device_buffer_capacity = capacity;
        }

        auto ptr = buffer_base + device_buffer_used;
        device_buffer_used += size;
        return ptr;
    };

    auto alloc_device_tensor = [&](ggml_tensor* tensor)
    {
        const auto size = get_aligned_size(ggml_backend_buft_get_alloc_size(buft, tensor), alignment);
        if (size > max_buffer_size)
            throw std::runtime_error(std::format(
                "Tensor '{}' needs {} bytes, which exceeds the maximum weight buffer size ({} bytes). "
                "The backend limit or COSYVOICE_MAX_WEIGHT_BUFFER_SIZE is too small for this model.\n",
                tensor->name, size, max_buffer_size));

        auto* ptr = alloc_device_bytes(size);
        ggml_backend_tensor_alloc(device_buffers.back().get(), tensor, ptr);
    };

    // The UMA probe uses the first buffer's base as scratch space, so it has to
    // run before the weights are uploaded.
    alloc_device_bytes(0);
    shared->backend_uma = backend_looks_uma(backend, device_buffers.front().get());

    shared->ctx.reset(ggml_init(params));

    mem_size = ggml_backend_buft_get_alloc_size(ggml_backend_cpu_buffer_type(), llm.embed_tokens_weight)
        + ggml_backend_buft_get_alloc_size(ggml_backend_cpu_buffer_type(), llm.speech_embedding_weight)
        + sizeof(float) * hift.nfft;
    shared->cpu_buffer.reset(ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(cpu_backend), mem_size));
    auto cpu_buffer_base = reinterpret_cast<char*>(ggml_backend_buffer_get_base(shared->cpu_buffer.get()));

    auto set_tensor = [&](ggml_tensor* tensor, const void* data, size_t size)
    {
        ggml_backend_tensor_set_async(backend, tensor, data, 0, size);
    };
    auto set_cpu_tensor = [&](ggml_tensor* tensor, const void* data, size_t size)
    {
        ggml_backend_tensor_set(tensor, data, 0, size);
        cpu_buffer_base += ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(cpu_backend), tensor);
    };
    if (!adaln_host.empty())
    {
        flow.decoder.estimator.adaln_table = ggml_new_tensor_3d(shared->ctx.get(), GGML_TYPE_F32, adaln_dim, adaln_rows, adaln_steps);
        ggml_set_name(flow.decoder.estimator.adaln_table, "dit.adaln_table");
        alloc_device_tensor(flow.decoder.estimator.adaln_table);
        set_tensor(flow.decoder.estimator.adaln_table, adaln_host.data(), adaln_host.size() * sizeof(float));
    }
    else
    {
        flow.decoder.estimator.time_embed.time_embed.emb = ggml_new_tensor_1d(shared->ctx.get(), GGML_TYPE_F32, half_dim);
        alloc_device_tensor(flow.decoder.estimator.time_embed.time_embed.emb);
        set_tensor(flow.decoder.estimator.time_embed.time_embed.emb, emb_buffer.get(), sizeof(float) * half_dim);
    }

    std::unordered_map<ggml_tensor**, std::string> tensor_name_map;
    tensor_name_map.reserve(tensors.size());
    for (const auto& [name, tptr] : tensors)
        tensor_name_map[tptr] = name;

    std::unordered_set<ggml_tensor**> fused_members;

    auto try_fuse_group = [&](std::initializer_list<ggml_tensor**> parts_list,
                              bool is_2d,
                              ggml_tensor** fused_out,
                              const std::string& fused_name,
                              ggml_tensor** tail_out = nullptr,
                              size_t tail_start = 1,
                              const std::string& tail_name = {})
    {
        std::vector<ggml_tensor**> parts(parts_list);
        if (parts.size() < 2)
            return;

        const ggml_type ref_type = (*parts[0]) ? (*parts[0])->type : GGML_TYPE_COUNT;
        const int64_t ref_ne0 = (*parts[0]) ? (*parts[0])->ne[0] : 0;
        int64_t total_cat = 0;
        size_t total_bytes = 0;
        for (size_t i = 0; i != parts.size(); ++i)
        {
            ggml_tensor* t = *parts[i];
            if (!t || fused_members.count(parts[i]) || get_tensor_eps(t) != 0.f)
                return;
            if (t->type != ref_type || t->ne[2] != 1 || t->ne[3] != 1)
                return;
            if (is_2d)
            {
                if (t->ne[0] != ref_ne0 || t->ne[1] <= 0)
                    return;
                total_cat += t->ne[1];
            }
            else
            {
                if (t->ne[1] != 1 || t->ne[0] <= 0)
                    return;
                total_cat += t->ne[0];
            }
            const size_t nbytes = ggml_nbytes(t);
            if (ggml_backend_buft_get_alloc_size(buft, t) != nbytes)
                return;
            if (i + 1 != parts.size() && (nbytes % alignment) != 0)
                return;
            total_bytes += nbytes;
        }

        const size_t aligned_total = get_aligned_size(total_bytes, alignment);
        if (aligned_total > max_buffer_size)
            return;

        ggml_tensor* fused = is_2d
            ? ggml_new_tensor_2d(shared->ctx.get(), ref_type, ref_ne0, total_cat)
            : ggml_new_tensor_1d(shared->ctx.get(), ref_type, total_cat);
        if (ggml_backend_buft_get_alloc_size(buft, fused) != total_bytes)
            return;
        ggml_set_name(fused, fused_name.c_str());

        char* base = alloc_device_bytes(aligned_total);
        auto* cur_buf = device_buffers.back().get();
        ggml_backend_tensor_alloc(cur_buf, fused, base);
        *fused_out = fused;

        size_t byte_off = 0;
        size_t tail_byte_off = 0;
        int64_t tail_cat = 0;
        for (size_t i = 0; i != parts.size(); ++i)
        {
            ggml_tensor* orig = *parts[i];
            const size_t nbytes = ggml_nbytes(orig);
            if (i == tail_start)
                tail_byte_off = byte_off;
            if (i >= tail_start)
                tail_cat += is_2d ? orig->ne[1] : orig->ne[0];

            ggml_tensor* new_tensor = ggml_new_tensor(shared->ctx.get(), orig->type, GGML_MAX_DIMS, orig->ne);
            if (auto it = tensor_name_map.find(parts[i]); it != tensor_name_map.end())
                ggml_set_name(new_tensor, it->second.c_str());
            ggml_backend_tensor_alloc(cur_buf, new_tensor, base + byte_off);
            set_tensor(new_tensor, ggml_get_data(orig), nbytes);
            *parts[i] = new_tensor;
            fused_members.insert(parts[i]);
            byte_off += nbytes;
        }

        if (tail_out && tail_cat > 0)
        {
            ggml_tensor* tail = is_2d
                ? ggml_new_tensor_2d(shared->ctx.get(), ref_type, ref_ne0, tail_cat)
                : ggml_new_tensor_1d(shared->ctx.get(), ref_type, tail_cat);
            if (ggml_backend_buft_get_alloc_size(buft, tail) == (total_bytes - tail_byte_off))
            {
                ggml_set_name(tail, tail_name.c_str());
                ggml_backend_tensor_alloc(cur_buf, tail, base + tail_byte_off);
                *tail_out = tail;
            }
        }
    };

    for (size_t i = 0; i != llm.layers.size(); ++i)
    {
        auto& attn = llm.layers[i].self_attn;
        try_fuse_group({ &attn.q_proj.weight, &attn.k_proj.weight, &attn.v_proj.weight }, true,
            &attn.qkv_weight, std::format("layers.{}.self_attn.qkv.weight", i));
        if (attn.qkv_weight && attn.q_proj.bias && attn.k_proj.bias && attn.v_proj.bias)
            try_fuse_group({ &attn.q_proj.bias, &attn.k_proj.bias, &attn.v_proj.bias }, false,
                &attn.qkv_bias, std::format("layers.{}.self_attn.qkv.bias", i));

        auto& mlp = llm.layers[i].mlp;
        if (!mlp.gate_proj.bias && !mlp.up_proj.bias)
            try_fuse_group({ &mlp.gate_proj.weight, &mlp.up_proj.weight }, true,
                &mlp.gate_up_weight, std::format("layers.{}.mlp.gate_up.weight", i));
    }

    for (const auto& [name, tensor] : tensors)
    {
        if (fused_members.count(tensor) || precomputed_adaln_tensors.count(tensor))
            continue;
        size_t tensor_size = ggml_nbytes(*tensor);
        auto new_tensor = ggml_new_tensor(shared->ctx.get(), (*tensor)->type, GGML_MAX_DIMS, (*tensor)->ne);
        ggml_set_name(new_tensor, name.c_str());
        if (tensor == &llm.embed_tokens_weight
            || tensor == &llm.speech_embedding_weight)
        {
            ggml_backend_tensor_alloc(shared->cpu_buffer.get(), new_tensor, cpu_buffer_base);
            set_cpu_tensor(new_tensor, ggml_get_data(*tensor), tensor_size);
        }
        else
        {
            alloc_device_tensor(new_tensor);

            const auto eps = get_tensor_eps(*tensor);
            if (eps == 0.f)
                set_tensor(new_tensor, ggml_get_data(*tensor), tensor_size);
            else
            {
                auto data = reinterpret_cast<const float*>(ggml_get_data(*tensor));
                const auto nelements = ggml_nelements(*tensor);
                std::unique_ptr<float[]> buffer;
                for (int64_t i = 0; i != nelements; ++i)
                    if (std::abs(data[i]) < eps)
                    {
                        if (!buffer)
                        {
                            buffer.reset(new float[nelements]);
                            memcpy(buffer.get(), data, sizeof(float) * nelements);
                        }

                        buffer[i] = eps;
                    }
                if (buffer)
                    ggml_backend_tensor_set(new_tensor, buffer.get(), 0, tensor_size);
                else
                    set_tensor(new_tensor, data, tensor_size);
            }
        }

        *tensor = new_tensor;
    }

    // Prepare HiFT Snake activations as [1, C] F32 (alpha, 1/alpha) pairs so
    // Vulkan fuses MUL->SIN->SQR->MUL->ADD into a single SNAKE kernel (72x).
    {
        ggml_backend_synchronize(backend);
        auto prepare_snake = [&](Snake& snk)
        {
            if (!snk.alpha || snk.alpha->type != GGML_TYPE_F32
                || snk.alpha->ne[1] != 1 || snk.alpha->ne[2] != 1 || snk.alpha->ne[3] != 1)
                return;
            const int64_t C = snk.alpha->ne[0];
            std::vector<float> vals(static_cast<size_t>(C));
            ggml_backend_tensor_get(snk.alpha, vals.data(), 0, vals.size() * sizeof(float));
            for (float& v : vals)
                v = 1.0f / v;

            // Reshape alpha in-place from [C] to [1, C] (identical contiguous layout, zero VIEW ops).
            snk.alpha->ne[0] = 1;
            snk.alpha->ne[1] = C;
            snk.alpha->nb[0] = sizeof(float);
            snk.alpha->nb[1] = sizeof(float);
            snk.alpha->nb[2] = static_cast<size_t>(C) * sizeof(float);
            snk.alpha->nb[3] = snk.alpha->nb[2];

            ggml_tensor* inv = ggml_new_tensor_2d(shared->ctx.get(), GGML_TYPE_F32, 1, C);
            ggml_set_name(inv, (std::string(snk.alpha->name) + ".inv").c_str());
            alloc_device_tensor(inv);
            ggml_backend_tensor_set(inv, vals.data(), 0, vals.size() * sizeof(float));
            snk.inv_alpha = inv;
        };
        auto prepare_resblocks = [&](std::vector<ResBlock>& blocks)
        {
            for (auto& rb : blocks)
                for (auto& [act1, conv1, act2, conv2] : rb.convs)
                {
                    prepare_snake(act1);
                    prepare_snake(act2);
                }
        };
        prepare_resblocks(hift.resblocks);
        prepare_resblocks(hift.source_resblocks);
    }

    hift.window = ggml_new_tensor_1d(shared->ctx.get(), GGML_TYPE_F32, hift.nfft);
    ggml_set_name(hift.window, "stft_window");
    ggml_backend_tensor_alloc(shared->cpu_buffer.get(), hift.window, cpu_buffer_base);
    for (int i = 0; i != hift.nfft; ++i)
        reinterpret_cast<float*>(hift.window->data)[i] = (1.0f - std::cos(2.0f * 3.14159265358979323846f * i / hift.nfft)) / 2.f;

    hift.fctx = create_fft_context(hift.nfft);
    hift.ictx = create_istft_context(hift.nfft, shared->ctx.get(),
        [&](ggml_tensor* tensor, void* data, size_t size)
        {
            alloc_device_tensor(tensor);
            set_tensor(tensor, data, size);
            ggml_backend_synchronize(backend);
        });

    shared->full_position_ids.reset(new int[shared->params.n_max_seq]);
    for (int i = 0; i != shared->params.n_max_seq; ++i)
        shared->full_position_ids.get()[i] = i;

    for (uint32_t i = 0; i != shared->params.n_workers; ++i)
    {
        auto worker = workers + i;

        worker->position_ids = ggml_new_tensor_1d(shared->ctx.get(), GGML_TYPE_I32, shared->params.n_batch);
        ggml_set_name(worker->position_ids, std::format("position_ids.{}", i).c_str());
        alloc_device_tensor(worker->position_ids);

        worker->causal_mask = ggml_new_tensor_2d(shared->ctx.get(), GGML_TYPE_F16, shared->params.n_max_seq - 1, shared->params.n_batch);
        ggml_set_input(worker->causal_mask);

        worker->causal_mask_task = ggml_new_tensor_2d(shared->ctx.get(), GGML_TYPE_F16, shared->params.n_max_seq - 1, shared->params.n_batch);
        ggml_set_input(worker->causal_mask_task);

        worker->decode_mask = ggml_new_tensor_2d(shared->ctx.get(), GGML_TYPE_F16, shared->params.n_max_seq, 1);
        ggml_set_name(worker->decode_mask, std::format("decode_mask.{}", i).c_str());
        alloc_device_tensor(worker->decode_mask);
    }

    shared->noise_rng.seed(shared->params.seed);

    hift.m_source.l_sin_gen.rand_ini = ggml_new_tensor_1d(shared->ctx.get(), GGML_TYPE_F32, hift.nb_harmonics + 1);
    alloc_device_tensor(hift.m_source.l_sin_gen.rand_ini);

    if (device_buffers.size() > 1)
        cosyvoice_call_ggml_log_callback(
            GGML_LOG_LEVEL_INFO,
            std::format(
                "weights: packed into {} buffers ({} MiB total, backend max {} MiB per buffer)\n",
                device_buffers.size(), device_total >> 20, max_buffer_size >> 20
            ).c_str());

    auto rand_ini_buffer = std::make_unique<float[]>(hift.nb_harmonics + 1);
    rand_ini_buffer[0] = 0.f;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& i : std::span(rand_ini_buffer.get() + 1, hift.nb_harmonics))
        i = dist(shared->noise_rng);
    hift.set_rand_ini(rand_ini_buffer.get());

    int64_t id = loader.parser.find_key("stop_token_ids");
    auto stop_tok_data = reinterpret_cast<const int*>(loader.parser.arr_data(id));
    id = static_cast<int64_t>(loader.parser.arr_n(id));
    cv3_shared->stop_tokens.insert(stop_tok_data, stop_tok_data + id);

    // FSQ silent and breath tokens �?loaded from GGUF if available,
    // otherwise fall back to hardcoded defaults matching CosyVoice3.
    id = loader.parser.find_key("silent_token_ids");
    if (id != -1)
    {
        auto silent_tok_data = reinterpret_cast<const int*>(loader.parser.arr_data(id));
        id = static_cast<int64_t>(loader.parser.arr_n(id));
        cv3_shared->silent_tokens.insert(silent_tok_data, silent_tok_data + id);
    }
    else
    {
        static const int kDefaultSilentTokens[] = { 1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323 };
        cv3_shared->silent_tokens.insert(std::begin(kDefaultSilentTokens), std::end(kDefaultSilentTokens));
    }

    id = loader.parser.find_key("cosyvoice.instruction_prefix");
    if (id != -1)
    {
        const auto str = loader.parser.val_str(id);
        shared->instruction_prefix.reset(new char[str.size() + 1]);
        std::memcpy(shared->instruction_prefix.get(), str.data(), str.size());
        shared->instruction_prefix.get()[str.size()] = '\0';
    }

    shared->config.temperature = 1.f;
    shared->config.max_token_text_ratio = 20.f;
    shared->config.min_token_text_ratio = 2.f;

    auto& sampling = shared->config.sampling;
    LOAD_METADATA_NOPREFIX(sampling.top_k);
    LOAD_METADATA_NOPREFIX(sampling.top_p);
    LOAD_METADATA_NOPREFIX(sampling.win_size);
    LOAD_METADATA_NOPREFIX(sampling.tau_r);

    for (uint32_t i = 0; i != shared->params.n_workers; ++i)
        workers[i].config = shared->config;

    auto [llm_k_type, llm_v_type] = cosyvoice_check_kv_cache_types(worker->ctx0.get(), worker->backend.get(),
        shared->params.llm_use_flash_attn, cv3_shared->llm.num_attention_heads, cv3_shared->llm.num_key_value_heads,
        cv3_shared->llm.layers[0].self_attn.q_proj, cv3_shared->llm.layers[0].self_attn.k_proj, cv3_shared->llm.layers[0].self_attn.v_proj,
        shared->params.llm_allow_kv_cache_fallback, reinterpret_cast<kv_cache_type_union&>(shared->params.llm_kv_cache_type));

    auto [dit_k_type, dit_v_type] = (shared->params.dit_kv_fixed_slots + shared->params.dit_kv_offloadable_slots) != 0 ?
        cosyvoice_check_kv_cache_types(worker->ctx0.get(), worker->backend.get(),
            shared->params.flow_use_flash_attn, cv3_shared->flow.decoder.estimator.transformer_blocks[0].attn.heads, cv3_shared->flow.decoder.estimator.transformer_blocks[0].attn.heads,
            cv3_shared->flow.decoder.estimator.transformer_blocks[0].attn.to_q, cv3_shared->flow.decoder.estimator.transformer_blocks[0].attn.to_k, cv3_shared->flow.decoder.estimator.transformer_blocks[0].attn.to_v,
            shared->params.dit_allow_kv_cache_fallback, reinterpret_cast<kv_cache_type_union&>(shared->params.dit_kv_cache_type))
        : std::tuple<ggml_type, ggml_type>();

    for (auto& worker : std::span(workers, shared->params.n_workers))
    {
        worker.chunk_size = 25 + cv3_shared->flow.pre_lookahead_layer.pre_lookahead_len;
        worker.nucleus_probs_capacity = static_cast<uint32_t>(sampling.top_k * 2);
        worker.nucleus_probs.reset(new float[worker.nucleus_probs_capacity]);
        worker.nucleus_raw.reset(new float[worker.nucleus_probs_capacity]);
        worker.nucleus_probs_len = 0;
        worker.probs.reset(new float[llm.llm_decoder.weight->ne[1]]);
        worker.batch_buffer.reset(new char[shared->params.n_batch * std::max(llm.embed_tokens_weight->nb[1], llm.speech_embedding_weight->nb[1])]);

        worker.llm_kv_cache.build_kv_cache(
            backend,
            worker.llm_kv_buffer,
            static_cast<int>(llm.layers.size()),
            static_cast<int>(llm.layers[0].self_attn.k_proj.weight->ne[1] / llm.num_key_value_heads),
            static_cast<int>(llm.layers[0].self_attn.v_proj.weight->ne[1] / llm.num_key_value_heads),
            llm.num_key_value_heads,
            shared->params.n_max_seq,
            llm_k_type,
            llm_v_type,
            1,
            1,
            shared->params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED ? 0 : 1,
            shared->params.llm_use_flash_attn
        );

        if (shared->params.dit_kv_fixed_slots + shared->params.dit_kv_offloadable_slots != 0)
        {
            const auto dit_blocks = flow.decoder.estimator.transformer_blocks;
            worker.dit_kv_cache.build_kv_cache(
                backend,
                worker.dit_kv_buffer,
                static_cast<int>(dit_blocks.size()),
                static_cast<int>(dit_blocks[0].attn.to_k.weight->ne[1] / dit_blocks[0].attn.heads),
                static_cast<int>(dit_blocks[0].attn.to_v.weight->ne[1] / dit_blocks[0].attn.heads),
                dit_blocks[0].attn.heads,
                shared->params.dit_kv_cache_length,
                dit_k_type,
                dit_v_type,
                2,
                shared->params.dit_kv_actual_fixed_slots + (shared->params.dit_kv_offloadable_slots != 0 ? 1 : 0),
                shared->params.dit_kv_actual_offloadable_slots,
                shared->params.flow_use_flash_attn
            );
        }
    }

    for (auto& block : flow.decoder.estimator.transformer_blocks)
        block.attn.fattn = shared->params.flow_use_flash_attn;

    for (uint32_t i = 0; i < shared->params.n_workers; ++i)
    {
        auto cv3_worker = cv3_workers + i;
        auto worker = workers + i;

        switch (shared->params.inference_buffer_policy)
        {
        case COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED:
        case COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED:
            cv3_worker->token2wav_buffer.reset(worker->llm_kv_buffer.get());
        case COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED:
            break;
        default:
            throw std::invalid_argument("unexpected policy");
        }
    }

    auto arch = loader.get_string("general.architecture");
    shared->architecture.reset(new char[arch.size() + 1]);
    std::memcpy(shared->architecture.get(), arch.data(), arch.size());
    shared->architecture.get()[arch.size()] = '\0';
    shared->hift_overlap = hift.overlap_length();

    ggml_backend_synchronize(backend);
}
