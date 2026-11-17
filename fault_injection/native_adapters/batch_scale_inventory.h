#ifndef DCAT_DQ_INDEPENDENT_INVENTORY_H
#define DCAT_DQ_INDEPENDENT_INVENTORY_H
// Independent bounded metadata. No owner, arm, numerical work, or device write.
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace dcat_dq_inventory {
enum Kind : uint32_t { PrepareEnter=1, PrepareExit=2, ExecuteEnter=3, ExecuteExit=4 };
struct Event {
    uint32_t version{1}, size{sizeof(Event)}, kind{}, flags{};
    int32_t device{-1}, status{};
    uint64_t sequence{}, call{}, thread{}, executor{}, input{}, quant{}, scale{};
    int64_t rows{}, columns{};
    int32_t input_dtype{-1}, quant_dtype{-1}, scale_dtype{-1};
    int32_t input_format{-1}, quant_format{-1}, scale_format{-1};
    uint64_t original_calls{};
};
struct State {
    uint32_t version{1}, size{sizeof(State)}, reserved1{}, reserved2{};
    uint64_t capacity{}, total_events{}, stored_events{}, overflow_events{}, errors{};
    uint64_t prepare_enter{}, prepare_exit{}, execute_enter{}, execute_exit{};
    uint64_t metadata_complete{}, orphan_executions{}, registrations{}, pending{}, executing{};
};
template <std::size_t Capacity> class Inventory {
    static_assert(std::is_trivially_copyable<Event>::value, "Wire event must be trivially copyable");
    std::mutex mutex_;
    State state_{};
    std::array<std::array<unsigned char,sizeof(Event)>,Capacity> records_{};
    std::unordered_map<uint64_t,Event> pending_;
    void append(Event& event) {
        event.sequence=++state_.total_events;
        switch(event.kind) {
        case PrepareEnter:++state_.prepare_enter;break;case PrepareExit:++state_.prepare_exit;break;
        case ExecuteEnter:++state_.execute_enter;break;case ExecuteExit:++state_.execute_exit;break;
        default:++state_.errors;
        }
        if(state_.stored_events<Capacity)
            std::memcpy(records_[state_.stored_events++].data(),&event,sizeof(Event));
        else ++state_.overflow_events;
    }
public:
    Inventory() { state_.capacity=Capacity; }
    Event begin(Event event) {
        std::lock_guard<std::mutex> lock(mutex_);
        event.kind=PrepareEnter;event.call=state_.total_events+1;
        if(event.flags==31)++state_.metadata_complete;
        append(event);return event;
    }
    void prepared(Event event,uint64_t executor,int status,uint64_t original_calls) {
        std::lock_guard<std::mutex> lock(mutex_);
        event.kind=PrepareExit;event.executor=executor;event.status=status;event.original_calls=original_calls;
        if(!status) {
            if(!executor || pending_.size()>=4096 || !pending_.emplace(executor,event).second)++state_.errors;
            else {++state_.registrations;++state_.pending;}
        }
        append(event);
    }
    Event enter(uint64_t executor,uint64_t thread) {
        std::lock_guard<std::mutex> lock(mutex_);
        Event event{};event.executor=executor;event.thread=thread;
        const auto found=pending_.find(executor);
        if(found==pending_.end() || found->second.kind==ExecuteEnter || found->second.thread!=thread) {
            ++state_.orphan_executions;
            if(found!=pending_.end())++state_.errors;
        } else {
            event=found->second;found->second.kind=ExecuteEnter;++state_.executing;
        }
        event.kind=ExecuteEnter;event.status=0;event.original_calls=0;append(event);return event;
    }
    void leave(Event event,int status,uint64_t original_calls) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found=pending_.find(event.executor);
        if(event.call) {
            if(found==pending_.end() || found->second.call!=event.call || found->second.kind!=ExecuteEnter)++state_.errors;
            else {pending_.erase(found);--state_.pending;--state_.executing;}
        }
        event.kind=ExecuteExit;event.status=status;event.original_calls=original_calls;append(event);
    }
    State state() { std::lock_guard<std::mutex> lock(mutex_);return state_; }
    bool read(uint64_t index,Event* output) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!output || index>=state_.stored_events)return false;
        std::memcpy(output,records_[index].data(),sizeof(Event));return true;
    }
};
}
#endif
