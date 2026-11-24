#include "cosyvoice-model.h"
#include <algorithm>
#include <cmath>
#include <span>
#include <cstdio>

extern bool aitable_adapter_read_owned(void* owner);
namespace {
constexpr size_t factor_count = 270336;
void invalidate(cosyvoice_model_3* model) {
    // All workers are idle under the exclusive adapter gate. No model weights
    // or noise state are changed; prefill and decode graphs must see the mode.
    for (uint32_t i=0; i<model->shared->params.n_workers; ++i) {
        auto& w=model->workers[i];
        ggml_backend_synchronize(w.backend.get());
        ggml_backend_sched_reset(w.sched.get());
        w.gf=nullptr; w.llm_input=nullptr; w.llm_probs=nullptr;
        w.llm_top_k=nullptr; w.llm_full_probs=nullptr;
        w.prompt_crc32=0; w.llm_kv_cache.cur_len=0;
        w.llm_kv_cache.clear_offloaded_cache();
    }
}
}
int cosyvoice_lora_load_f32(cosyvoice_context* ctx,const float* data,size_t count) {
    auto* model=dynamic_cast<cosyvoice_model_3*>(ctx);
    if (!model || !data || count!=factor_count) return 3;
    auto& state=*model->cv3_shared;
    std::unique_lock gate(model->shared->adapter_mutex,std::try_to_lock);
    if (!gate.owns_lock()) return 1;
    if (state.lora_buffer || state.llm.layers.size()!=24) return 2;
    if (!std::all_of(data,data+count,[](float f){return std::isfinite(f);})) return 3;
    for (auto& layer:state.llm.layers) {
        auto* q=layer.self_attn.q_proj.weight;auto* v=layer.self_attn.v_proj.weight;
        if (!q || !v || q->type!=GGML_TYPE_F16 || v->type!=GGML_TYPE_F16 ||
            q->ne[0]!=896 || q->ne[1]!=896 || v->ne[0]!=896 || v->ne[1]!=128) return 2;
    }
    try {
        ggml_context_ptr storage(ggml_init({ggml_tensor_overhead()*100, nullptr, true}));
        if (!storage) return 4;
        std::vector<std::pair<ggml_tensor*,ggml_tensor*>> tensors;
        for (int i=0;i<24;++i) for(int output:{896,128}) {
            tensors.emplace_back(ggml_new_tensor_2d(storage.get(),GGML_TYPE_F32,896,4),
                                 ggml_new_tensor_2d(storage.get(),GGML_TYPE_F32,4,output));
        }
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(storage.get(),model->worker->backend.get()));
        if (!buffer) return 4;
        size_t offset=0;
        for (auto [a,b]:tensors) {
            for (auto* tensor:{a,b}) {
                auto bytes=ggml_nbytes(tensor);
                ggml_backend_tensor_set(tensor,data+offset,0,bytes); offset+=bytes/sizeof(float);
            }
        }
        // Commit only after every tensor is allocated and uploaded.
        state.lora_ctx=std::move(storage);state.lora_buffer=std::move(buffer);
        size_t index=0;
        for(auto& layer:state.llm.layers) for(auto* linear:{&layer.self_attn.q_proj,&layer.self_attn.v_proj}) {
            auto [a,b]=tensors[index++];linear->lora_a=a;linear->lora_b=b;
            linear->lora_enabled=&state.lora_enabled;
        }
        return 0;
    } catch(const std::bad_alloc&) {return 4;}
}
int cosyvoice_lora_set_enabled(cosyvoice_context* ctx,bool enabled) {
    auto* model=dynamic_cast<cosyvoice_model_3*>(ctx);if(!model)return 2;
    std::unique_lock gate(model->shared->adapter_mutex,std::try_to_lock);
    if(!gate.owns_lock())return 1;
    auto& state=*model->cv3_shared;
    if(enabled && !state.lora_buffer)return 3;
    if(enabled==state.lora_enabled)return 0;
    invalidate(model);
    state.lora_enabled=enabled;++state.lora_generation;
    return 0;
}
int cosyvoice_lora_state(cosyvoice_context* ctx,uint64_t* generation,size_t* bytes) {
    auto* model=dynamic_cast<cosyvoice_model_3*>(ctx);if(!model)return -1;
    std::shared_lock<std::shared_mutex> gate;
    if(!aitable_adapter_read_owned(model->shared))gate=std::shared_lock(model->shared->adapter_mutex);
    auto& state=*model->cv3_shared;
    if(generation)*generation=state.lora_generation;
    if(bytes)*bytes=state.lora_buffer?ggml_backend_buffer_get_size(state.lora_buffer.get()):0;
    return !state.lora_buffer?0:state.lora_enabled?2:1;
}
