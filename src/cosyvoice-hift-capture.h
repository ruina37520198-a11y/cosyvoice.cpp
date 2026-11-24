#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
namespace hift_research {
// Request-owned Level 3 capture. The host reduces native level below 3 when
// diagnostic privacy is enabled or unknown. There is no implicit capture path.
inline constexpr size_t total_limit=512ull*1024*1024, request_limit=128ull*1024*1024;
struct Request {
    std::filesystem::path root;
    unsigned calls=0;
    size_t bytes=0;
    bool failed=false;
};
inline std::mutex registry_mutex;
inline std::map<const void*,std::shared_ptr<Request>> requests;
inline std::atomic<unsigned long long> serial{0};
inline size_t reserved_total=0;
inline bool inventory_loaded=false;
inline std::filesystem::path inventory_root;
inline void failed(const std::shared_ptr<Request>& request,const char* code) noexcept {
    if (request && request->failed) return;
    if (request) request->failed=true;
    std::fprintf(stderr,"[AITABLE_HIFT_CAPTURE] status=unavailable reason=%s\n",code);
}
inline std::shared_ptr<Request> request(const void* owner,bool fresh,bool enabled) noexcept {
    if (!enabled) return {};
    const char* path=std::getenv("AITABLE_HIFT_CAPTURE_DIR");
    if (!path || !*path) return {};
    try {
        std::lock_guard lock(registry_mutex);
        if (!fresh) {
            auto it=requests.find(owner);
            return it==requests.end()?nullptr:it->second;
        }
        requests.erase(owner);
        if (requests.size()>=64) { failed({},"request_capacity"); return {}; }
        const std::filesystem::path root(path);
        if (!inventory_loaded) {
            std::filesystem::create_directories(root);
            size_t entries=0, existing_bytes=0;
            for (const auto& entry:std::filesystem::recursive_directory_iterator(root)) {
                if (++entries>8192) { failed({},"inventory_capacity"); return {}; }
                if (entry.is_symlink()) { failed({},"unsafe_capture_entry"); return {}; }
                if (entry.is_regular_file()) {
                    const auto size=entry.file_size();
                    if (size>total_limit || existing_bytes>total_limit-size) { failed({},"storage_capacity"); return {}; }
                    existing_bytes+=static_cast<size_t>(size);
                }
            }
            reserved_total=existing_bytes;inventory_loaded=true;inventory_root=root;
        }
        if (inventory_root!=root || reserved_total>=total_limit) { failed({},"storage_capacity"); return {}; }
        auto value=std::make_shared<Request>();
        const auto stamp=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        value->root=root/("request-"+std::to_string(stamp)+"-"+std::to_string(serial.fetch_add(1)));
        if (!std::filesystem::create_directory(value->root)) { failed({},"directory_collision"); return {}; }
        requests.emplace(owner,value);
        return value;
    } catch (...) { failed({},"storage_unavailable"); return {}; }
}
inline void finish(const void* owner,bool final) noexcept {
    if (!final) return;
    std::lock_guard lock(registry_mutex);requests.erase(owner);
}
struct Capture {
    std::shared_ptr<Request> owner;
    unsigned index=0,seed=0,worker=0;
    bool complete=false;
    std::vector<std::pair<std::string,ggml_tensor*>> tensors;
    explicit Capture(std::shared_ptr<Request> value,unsigned seed,unsigned worker):owner(std::move(value)),seed(seed),worker(worker) {
        if (!owner || owner->failed) { owner.reset();return; }
        index=owner->calls++;
        if (index>=8) { failed(owner,"request_call_capacity");owner.reset(); }
    }
    void keep(const char* name,ggml_tensor* tensor) noexcept {
        if (!owner || owner->failed) return;
        try {
            const size_t n=ggml_nbytes(tensor), overhead=1024;
            std::lock_guard lock(registry_mutex);
            if (n>request_limit-overhead || owner->bytes>request_limit-n-overhead || reserved_total>total_limit-n-overhead) {
                failed(owner,"tensor_capacity");return;
            }
            owner->bytes+=n+overhead;reserved_total+=n+overhead;
            tensors.emplace_back(name,tensor);
            ggml_set_output(tensor);
        } catch (...) { failed(owner,"capture_allocation"); }
    }
    void write() noexcept {
        if (!owner || owner->failed) return;
        try {
            for (const auto& [name,tensor]:tensors) {
                std::vector<char> data(ggml_nbytes(tensor));
                ggml_backend_tensor_get(tensor,data.data(),0,data.size());
                auto stem=owner->root/("call"+std::to_string(index)+"."+name);
                std::ofstream file;file.exceptions(std::ios::failbit|std::ios::badbit);
                file.open(stem.string()+".bin",std::ios::binary|std::ios::trunc);
                file.write(data.data(),static_cast<std::streamsize>(data.size()));file.close();
                std::ofstream meta;meta.exceptions(std::ios::failbit|std::ios::badbit);meta.open(stem.string()+".json");
                meta<<"{\"type\":"<<tensor->type<<",\"ne\":[";
                for (int i=0;i<4;++i) meta<<(i?",":"")<<tensor->ne[i];
                meta<<"],\"nb\":[";
                for (int i=0;i<4;++i) meta<<(i?",":"")<<tensor->nb[i];
                meta<<"],\"bytes\":"<<data.size()<<"}";meta.close();
            }
            complete=true;
        } catch (...) { failed(owner,"save_failed"); }
    }
    void boundary(uint32_t prior,uint32_t end,uint32_t emitted,bool final,int64_t cache_frames,int64_t new_frames) noexcept {
        if (!owner || owner->failed || !complete) return;
        try {
            std::ofstream out;out.exceptions(std::ios::failbit|std::ios::badbit);
            out.open(owner->root/("call"+std::to_string(index)+".boundary.json"));
            out<<"{\"protocol\":1,\"seed\":"<<seed<<",\"worker\":"<<worker<<",\"prior_commit\":"<<prior<<",\"new_commit\":"<<end
               <<",\"emitted_samples\":"<<emitted<<",\"finalize\":"<<(final?"true":"false")<<",\"cache_frames\":"<<cache_frames
               <<",\"new_frames\":"<<new_frames<<",\"tensor_capture_complete\":true}";out.close();
            std::fprintf(stderr,"[AITABLE_HIFT_CAPTURE] status=saved call=%u seed=%u worker=%u emitted=%u final=%u\n",index,seed,worker,emitted,final?1:0);
        } catch (...) { failed(owner,"save_failed"); }
    }
};
}
