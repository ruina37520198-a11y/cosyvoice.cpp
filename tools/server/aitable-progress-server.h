// Bounded metadata only. Never holds the registry mutex during inference or I/O.
#pragma once
#include <array>
#include <memory>
#include <mutex>
#include <deque>
#include <atomic>
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif
#if defined(_WIN32)
#define AITABLE_DIAG_PID _getpid
#else
#define AITABLE_DIAG_PID getpid
#endif
namespace aitable_progress {
inline uint64_t now_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct record {
    std::string token;
    uint32_t pid,seed;
    cosyvoice_context_t context;
    std::atomic<bool> active{true};
    std::atomic<uint64_t> sequence{0};
    std::array<std::atomic<uint64_t>,16> values{};
    record(std::string t,uint32_t s,cosyvoice_context_t ctx):token(std::move(t)),pid(static_cast<uint32_t>(AITABLE_DIAG_PID())),seed(s),context(ctx){}
    static void observe(const cosyvoice_progress_t* p,void* data) noexcept {
        auto& r=*static_cast<record*>(data);
        r.sequence.fetch_add(1,std::memory_order_acq_rel);
        cosyvoice_failure_stats_t failure{};
        cosyvoice_get_failure_stats(r.context,1,sizeof(failure),&failure);
        const uint64_t fields[]={p->stage,p->phase,p->worker,p->seed,p->thread_id,p->samples,p->accepted,p->rejected_streak,p->kv_length,p->clock_ms,p->cancelled?1u:0u,failure.reason,failure.nucleus_positive,failure.nucleus_non_silent,failure.raw_non_silent_mass_ppb,failure.rescue_draws};
        for(size_t i=0;i<r.values.size();++i)r.values[i].store(fields[i],std::memory_order_relaxed);
        r.sequence.fetch_add(1,std::memory_order_release);
    }
    json snapshot() const {
        for(int attempt=0;attempt<8;++attempt){
            auto before=sequence.load(std::memory_order_acquire);if(before&1)continue;
            std::array<uint64_t,16> v{};for(size_t i=0;i<v.size();++i)v[i]=values[i].load(std::memory_order_relaxed);
            if(before!=sequence.load(std::memory_order_acquire))continue;
            static constexpr const char* stages[]={"unknown","request_ready","sampling","phase_begin","phase_end","compute_begin","compute_end","sync_begin","sync_end"};
            static constexpr const char* phases[]={"unknown","prefill","decode","sampling","llm"};
            json result={{"schema","aitable.cosy_progress@1"},{"request_token",token},{"pid",pid},{"seed",seed},{"active",active.load()},
                {"stage",v[0]<9?stages[v[0]]:"unknown"},{"phase",v[1]<5?phases[v[1]]:"unknown"},{"worker",v[2]},{"native_seed",v[3]},
                {"thread_id",v[4]},{"samples",v[5]},{"accepted",v[6]},{"rejected_streak",v[7]},{"kv_length",v[8]},
                {"last_progress_ms",v[9]},{"age_ms",v[9]?now_ms()-v[9]:0},{"cancelled",v[10]!=0},{"sequence",before}};
            if(v[11]==1 || v[11]==2)result["failure"]={{"reason",v[11]==1?"silent_resample_exhausted":"generated_token_limit"},
                {"nucleus_positive",v[12]},{"nucleus_non_silent",v[13]},
                {"raw_non_silent_mass_ppb",v[14]},{"rescue_draws",v[15]}};
            return result;
        }
        return {{"status","snapshot_busy"}};
    }
};
inline std::mutex records_mutex;
inline std::deque<std::shared_ptr<record>> records;
inline bool valid_token(const std::string& s){return s.size()==32 && s.find_first_not_of("0123456789abcdef")==std::string::npos;}
inline std::shared_ptr<record> begin(const Request& req,Response& res,cosyvoice_context_t ctx,uint32_t seed){
    const auto token=req.get_header_value("X-AiTable-Trace");
    if(!valid_token(token))return {};
    std::shared_ptr<record> r;
    {
        std::lock_guard guard(records_mutex);
        for(const auto& old:records)if(old->token==token)return {};
        if(records.size()>=32){
            auto old=std::find_if(records.begin(),records.end(),[](const auto& row){return !row->active.load();});
            if(old==records.end())return {};records.erase(old);
        }
        r=std::make_shared<record>(token,seed,ctx);records.push_back(r);
    }
    cosyvoice_set_progress_callback(ctx,record::observe,r.get());
    res.set_header("X-AiTable-Trace",token);
    res.set_header("X-AiTable-Pid",std::to_string(r->pid));
    res.set_header("X-AiTable-Seed",std::to_string(seed));
    res.set_header("X-AiTable-Worker",std::to_string(r->values[2].load()));
    return r;
}
inline void end(const std::shared_ptr<record>& r,cosyvoice_context_t ctx){
    if(r){cosyvoice_set_progress_callback(ctx,nullptr,nullptr);r->active.store(false);}
}
inline void route(Server& server){
    server.Get(R"(/aitable/progress/([0-9a-f]{32}))",[](const Request& req,Response& res){
        if(req.remote_addr!="127.0.0.1" && req.remote_addr!="::1"){res.status=403;return;}
        std::shared_ptr<record> selected;
        {
            std::lock_guard guard(records_mutex);
            for(auto& row:records)if(row->token==req.matches[1].str()){selected=row;break;}
        }
        if(!selected){res.status=404;return;}
        auto payload=selected->snapshot();res.status=payload.contains("schema")?200:503;
        res.set_header("Cache-Control","no-store");res.set_content(payload.dump(),"application/json");
    });
}
}
