#include "fft.h"
#include "ggml-fft.h"
#include "ggml-cpu-flag.h"

#include "simd-kernels.h"

#include <cmath>
#include <cstring>
#include <atomic>
#include <functional>

struct fft_context
{
    fft_context(int nfft);
    ~fft_context() = default;

    int factors[63];
    int max_factor_value;
    int nfft;
    std::atomic_uint workers;
    std::atomic_int64_t idx;
    std::unique_ptr<float[]> twiddles_r;
    float* twiddles_i;
};

// Runs the whole recursive Cooley-Tukey tree with ONE dispatch: the butterfly
// kernels are instantiated inside the tier object (see fft_work_kernel), so
// per-node dispatch and cross-TU calls are gone.
inline
static void kf_work(
    float* Foutr,
    float* Fouti,
    const float* f,
    int fstride,
    int in_stride,
    int* factors,
    float* buffer_scratchr,
    float* buffer_scratchi,
    fft_context& ctx
)
{
    simd_dispatch<fft_work_kernel>(Foutr, Fouti, f, fstride, in_stride, factors,
        buffer_scratchr, buffer_scratchi, ctx.twiddles_r.get(), ctx.twiddles_i, ctx.nfft);
}

fft_context::fft_context(int nfft) : nfft(nfft), idx(0), workers(0), twiddles_r(std::make_unique<float[]>(nfft * 2))
{
    twiddles_i = twiddles_r.get() + nfft;

    constexpr float pi = 3.1415926535897932f;
    const float factor = -2 * pi / nfft;
    simd_dispatch<init_twiddles_kernel>(twiddles_r.get(), twiddles_i, nfft, factor);
    int p = 4;
    auto n = nfft;
    double floor_sqrt = floor(sqrt(n));
    auto facbuf = factors;
    max_factor_value = 0;

    /*factor out powers of 4, powers of 2, then any remaining primes */
    do {
        while (n % p) {
            switch (p) {
            case 4: p = 2; break;
            case 2: p = 3; break;
            default: p += 2; break;
            }
            if (p > floor_sqrt)
                p = n;          /* no more factors, skip to end */
        }
        n /= p;
        *facbuf++ = p;
        *facbuf++ = n;
        if (p > max_factor_value)
            max_factor_value = p;
        if (n > max_factor_value)
            max_factor_value = n;
    } while (n > 1);
}

static void ggml_fft_op(struct ggml_tensor* dst, int ith, int nth, fft_context* ctx)
{
    auto buffer_scratchr = reinterpret_cast<float*>(alloca(sizeof(float) * ctx->max_factor_value * 2));
    auto buffer_scratchi = buffer_scratchr + ctx->max_factor_value;
    auto const signal = dst->src[0];
    auto const data = reinterpret_cast<char*>(signal->data);
    const auto foutr = reinterpret_cast<char*>(dst->data);
    const auto fouti = foutr + dst->nb[2];
    const auto dst_nb1 = dst->nb[1];
    const auto signal_nb1 = signal->nb[1];
    const auto signal_ne1 = signal->ne[1];

    const int64_t i0 = (signal_ne1 * ith) / nth;
    const int64_t i1 = (signal_ne1 * (ith + 1)) / nth;
    for (int64_t i = i0; i < i1; ++i)
    {
        const auto cur_offset = dst_nb1 * i;
        const auto cur_fouti = reinterpret_cast<float*>(fouti + cur_offset);
        memset(cur_fouti, 0, dst_nb1);
        kf_work(reinterpret_cast<float*>(foutr + cur_offset),
            cur_fouti,
            reinterpret_cast<float*>(data + signal_nb1 * i),
            1, 1, ctx->factors,
            buffer_scratchr, buffer_scratchi, *ctx);
    }
}

static void ggml_stft_fused_op(struct ggml_tensor* dst, int ith, int nth, fft_context* ctx)
{
    const auto* signal = dst->src[0];
    const auto* window = dst->src[1];
    const auto* meta   = dst->src[2];
    const int64_t hop_len = meta->op_params[4];
    const int64_t pad     = meta->op_params[5];

    const int nfft = ctx->nfft;
    const int n_bins = nfft / 2 + 1;
    const int64_t n_samples = signal->ne[0];
    const int64_t n_frames  = dst->ne[0];

    const float* sig_data = reinterpret_cast<const float*>(signal->data);
    const float* win_data = reinterpret_cast<const float*>(window->data);
    float* out_r = reinterpret_cast<float*>(dst->data);
    float* out_i = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + dst->nb[2]);

    float* win_frame = reinterpret_cast<float*>(alloca(sizeof(float) * (nfft * 3 + ctx->max_factor_value * 2)));
    float* foutr = win_frame + nfft;
    float* fouti = foutr + nfft;
    float* scratchr = fouti + nfft;
    float* scratchi = scratchr + ctx->max_factor_value;

    const int64_t f0 = (n_frames * ith) / nth;
    const int64_t f1 = (n_frames * (ith + 1)) / nth;
    for (int64_t f = f0; f < f1; ++f)
    {
        const int64_t base = f * hop_len - pad;
        if (base >= 0 && base + nfft <= n_samples)
        {
            const float* src = sig_data + base;
            for (int k = 0; k < nfft; ++k)
                win_frame[k] = src[k] * win_data[k];
        }
        else
        {
            for (int k = 0; k < nfft; ++k)
            {
                int64_t idx = base + k;
                if (idx < 0)
                    idx = -idx;
                else if (idx >= n_samples)
                    idx = 2 * (n_samples - 1) - idx;
                win_frame[k] = sig_data[idx] * win_data[k];
            }
        }

        memset(fouti, 0, sizeof(float) * nfft);
        kf_work(foutr, fouti, win_frame, 1, 1, ctx->factors, scratchr, scratchi, *ctx);

        for (int k = 0; k < n_bins; ++k)
        {
            const int64_t off = static_cast<int64_t>(k) * n_frames + f;
            out_r[off] = foutr[k];
            out_i[off] = fouti[k];
        }
    }
}

void delete_fft_context(fft_context* ctx)
{
    delete ctx;
}

fft_context_ptr create_fft_context(int nfft)
{
    return fft_context_ptr(new fft_context(nfft));
}

void fft(const float* signal, float* buffer, fft_context& ctx)
{
    auto nfft = ctx.nfft;
    auto foutr = reinterpret_cast<float*>(alloca(sizeof(float) * nfft * 2 + sizeof(float) * ctx.max_factor_value * 2));
    auto fouti = foutr + nfft;
    auto buffer_scratchr = fouti + nfft;
    auto buffer_scratchi = buffer_scratchr + ctx.max_factor_value;
    memset(fouti, 0, sizeof(float) * nfft);

    kf_work(foutr, fouti, signal, 1, 1, ctx.factors, buffer_scratchr, buffer_scratchi, ctx);
    simd_dispatch<fft_abs_kernel>(foutr, fouti, buffer, nfft);
}

ggml_tensor* ggml_fft(ggml_context* ctx, ggml_tensor* a, fft_context* fctx)
{
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(a->ne[0] == fctx->nfft);
    GGML_ASSERT(ggml_is_contiguous(a));

    return ggml_custom_4d(ctx, a->type, fctx->nfft, a->ne[1], 2, 1, &a, 1, reinterpret_cast<ggml_custom_op_t>(ggml_fft_op), GGML_N_TASKS_MAX, fctx);
}

ggml_tensor* ggml_stft(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int hop_len, bool center, fft_context* fctx)
{
    GGML_ASSERT(ggml_is_vector(a));
    GGML_ASSERT(ggml_is_vector(b));
    GGML_ASSERT(ggml_is_contiguous(a));

    const auto win_size = b->ne[0];
    const int pad_amount = center ? fctx->nfft / 2 : 0;
    if (win_size == fctx->nfft && hop_len > 0 && (!center || a->ne[0] > pad_amount))
    {
        const int64_t padded_len = a->ne[0] + 2 * pad_amount;
        const int64_t n_frames = (padded_len - win_size) / hop_len + 1;
        auto* meta = ggml_view_1d(ctx, b, 1, 0);
        meta->op_params[4] = hop_len;
        meta->op_params[5] = pad_amount;
        ggml_set_cpu(meta);
        ggml_tensor* args[] = { a, b, meta };
        return ggml_custom_4d(
            ctx, a->type, n_frames, fctx->nfft / 2 + 1, 2, 1,
            args, 3,
            reinterpret_cast<ggml_custom_op_t>(ggml_stft_fused_op),
            GGML_N_TASKS_MAX, fctx);
    }

    if (center)
        a = ggml_pad_reflect_1d(ctx, a, pad_amount, pad_amount);
    else
        a = ggml_dup(ctx, a);

    ggml_set_cpu(a);

    const auto n_frames = (a->ne[0] - win_size) / hop_len + 1;

    a = ggml_dup_inplace(ctx, a);
    a->ne[0] = win_size;
    a->ne[1] = n_frames;
    a->nb[1] = hop_len * a->nb[0];
    a->op = GGML_OP_VIEW;
    ggml_set_cpu(a);

    b = ggml_repeat(ctx, b, a);
    ggml_set_cpu(b);
    a = ggml_mul(ctx, a, b);
    ggml_set_cpu(a);

    auto result = ggml_fft(ctx, a, fctx);
    result = ggml_view_3d(ctx, result, fctx->nfft / 2 + 1, n_frames, 2, result->nb[1], result->nb[2], 0);
    return ggml_permute(ctx, result, 1, 0, 2, 3);
}

struct istft_context
{
    int nfft;
    ggml_tensor* idft_real;
    ggml_tensor* idft_imag;
};

void delete_istft_context(istft_context* ctx)
{
    delete ctx;
}

istft_context_ptr create_istft_context(int nfft, ggml_context* ctx, std::function<void(ggml_tensor*, void*, size_t)> set_data)
{
    const int f = nfft / 2 + 1;
    const int stride = f * nfft;
    auto idft_real = std::make_unique<float[]>(stride * 2);
    auto idft_imag = idft_real.get() + stride;
    for (int y = 0; y != f; ++y)
        for (int x = 0; x != nfft; ++x)
        {
            const float phase = 2.0f * 3.14159265358979323846f * x * y / nfft;
            const int offset = x * f + y;
            idft_real[offset] = std::cos(phase);
            idft_imag[offset] = phase;
        }

    for (int x = 0, y = f - 1; x != nfft; ++x)
    {
        const auto offset1 = x * f;
        const auto offset2 = offset1 + y;
        idft_real[offset1] /= nfft;
        idft_real[offset2] /= nfft;
        idft_imag[offset2] = 0.0f;
    }

    for (int y = 1, endy = f - 1; y != endy; ++y)
        for (int x = 0; x != nfft; ++x)
        {
            const int idx = x * f + y;
            auto& real = idft_real[idx];
            auto& imag = idft_imag[idx];

            real = 2.0f * real / nfft;
            imag = -2.0f * std::sin(imag) / nfft;
        }

    auto ictx = new istft_context;
    ictx->nfft = nfft;
    ictx->idft_real = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, f, nfft);
    ictx->idft_imag = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, f, nfft);
    set_data(ictx->idft_real, idft_real.get(), sizeof(float) * stride);
    set_data(ictx->idft_imag, idft_imag, sizeof(float) * stride);
    return istft_context_ptr(ictx);
}



static void ggml_istft_finalize_op(struct ggml_tensor* dst, int ith, int nth, void* userdata)
{
    static_assert(sizeof(uintptr_t) >= 8, "ggml_istft_finalize_op packs hop_len and offset into userdata");
    const auto packed = reinterpret_cast<uintptr_t>(userdata);
    const int64_t hop_len = static_cast<int64_t>(packed >> 32);
    const int64_t offset  = static_cast<int64_t>(packed & 0xFFFFFFFFu);

    auto* frames_t = dst->src[0];
    auto* win_t    = dst->src[1];
    auto* denom_t  = dst->src[2];

    const float* frames = reinterpret_cast<const float*>(frames_t->data);
    const float* win    = reinterpret_cast<const float*>(win_t->data);
    float* dst_data     = reinterpret_cast<float*>(dst->data);
    float* denom_data   = reinterpret_cast<float*>(denom_t->data);

    const int64_t out_len  = dst->ne[0];
    const int64_t win_len  = win_t->ne[0];
    const int64_t n_frames = frames_t->ne[1];

    float* w_sq = reinterpret_cast<float*>(alloca(sizeof(float) * static_cast<size_t>(win_len)));
    for (int64_t k = 0; k != win_len; ++k)
        w_sq[k] = win[k] * win[k];

    const int64_t begin = (out_len * ith) / nth;
    const int64_t end   = (out_len * (ith + 1)) / nth;
    if (begin >= end)
        return;

    memset(dst_data + begin, 0, static_cast<size_t>(end - begin) * sizeof(float));
    memset(denom_data + begin, 0, static_cast<size_t>(end - begin) * sizeof(float));

    const int64_t min_num = begin + offset - win_len + 1;
    int64_t f_start = min_num > 0 ? (min_num / hop_len) : 0;
    while (f_start > 0 && (f_start - 1) * hop_len + win_len - 1 - offset >= begin)
        --f_start;
    while (f_start < n_frames && f_start * hop_len + win_len - 1 - offset < begin)
        ++f_start;

    int64_t f_end = (end + offset - 1) >= 0 ? ((end + offset - 1) / hop_len + 1) : 0;
    if (f_end > n_frames)
        f_end = n_frames;

    for (int64_t f = f_start; f < f_end; ++f)
    {
        const float* fr = frames + f * win_len;
        const int64_t base = f * hop_len - offset;
        const int64_t k0 = base < begin ? (begin - base) : 0;
        const int64_t k1 = (base + win_len) > end ? (end - base) : win_len;
        for (int64_t k = k0; k < k1; ++k)
        {
            dst_data[base + k] += fr[k];
            denom_data[base + k] += w_sq[k];
        }
    }

    simd_dispatch<safe_divide_kernel>(dst_data + begin, denom_data + begin, static_cast<size_t>(end - begin), 1e-8f);
}

ggml_tensor* ggml_istft(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, ggml_tensor* c, int hop_len, bool center, istft_context* ictx)
{
    GGML_ASSERT(ggml_are_same_shape(a, b));
    GGML_ASSERT(a->ne[1] == ictx->nfft / 2 + 1);
    GGML_ASSERT(hop_len > 0);

    a = ggml_cont(ctx, ggml_permute(ctx, a, 1, 0, 2, 3));
    b = ggml_cont(ctx, ggml_permute(ctx, b, 1, 0, 2, 3));

    auto frames = ggml_add(ctx,
        ggml_mul_mat(ctx, ictx->idft_real, a),
        ggml_mul_mat(ctx, ictx->idft_imag, b));
    frames = ggml_mul(ctx, frames, c);

    auto out_len_full = (b->ne[1] - 1) * hop_len + (center ? 0 : ictx->nfft);
    const uintptr_t packed = (static_cast<uintptr_t>(static_cast<uint32_t>(hop_len)) << 32)
        | static_cast<uintptr_t>(static_cast<uint32_t>(center ? ictx->nfft / 2 : 0));
    ggml_tensor* args[] = { frames, c, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_len_full) };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, out_len_full, 1, 1, 1, args, 3, ggml_istft_finalize_op, GGML_N_TASKS_MAX, reinterpret_cast<void*>(packed));
}
