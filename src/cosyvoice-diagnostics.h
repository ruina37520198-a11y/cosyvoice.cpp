// Request-owned diagnostic budget. No TLS: native jobs can cross threads.
#pragma once
#include <atomic>
#include <cstdio>
#include <chrono>
namespace aitable_tts_diag {
struct state {
    std::atomic<unsigned> level{0};
    std::atomic<unsigned> remaining{0};
    void configure(unsigned value) noexcept {
        remaining.store(value >= 2 ? 128 : 0, std::memory_order_relaxed);
        level.store(value, std::memory_order_release);
    }
    bool enabled() const noexcept { return level.load(std::memory_order_acquire) >= 2; }
    bool take() noexcept {
        if (!enabled()) return false;
        unsigned n=remaining.load(std::memory_order_relaxed);
        while (n && !remaining.compare_exchange_weak(n,n-1,std::memory_order_relaxed)) {}
        return n!=0;
    }
};
inline void mark(state& owner,const char* stage,int count=-1,int worker=-1) noexcept {
    // Return before reading the clock, formatting, counters or stdio.
    if (!owner.take()) return;
    const auto ms=static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    char line[192];
    if (count>=0 && worker>=0)
        std::snprintf(line,sizeof(line),"[AITABLE_TTS_STAGE] stage=%s count=%d worker=%d clock_ms=%llu\n",stage,count,worker,ms);
    else if (worker>=0)
        std::snprintf(line,sizeof(line),"[AITABLE_TTS_STAGE] stage=%s worker=%d clock_ms=%llu\n",stage,worker,ms);
    else
        std::snprintf(line,sizeof(line),"[AITABLE_TTS_STAGE] stage=%s clock_ms=%llu\n",stage,ms);
    std::fputs(line,stderr);std::fflush(stderr);
}
struct scope {
    state& owner;const char* end;int worker;
    scope(state& s,const char* begin,const char* end,int worker=-1) noexcept:owner(s),end(end),worker(worker) {mark(owner,begin,-1,worker);}
    ~scope(){mark(owner,end,-1,worker);}
};
}
