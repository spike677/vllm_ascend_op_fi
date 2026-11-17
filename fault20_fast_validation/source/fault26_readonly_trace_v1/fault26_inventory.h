#ifndef DCAT26_INDEPENDENT_INVENTORY_H
#define DCAT26_INDEPENDENT_INVENTORY_H
// CPU metadata only. No callback, owner, hit, arm, allocation or device writes.
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace dcat26_inventory {
enum Kind : uint32_t {
    QmmEnter=1, QmmExit=2, FiaBefore=3, FiaGws=4, FiaEnter=5, FiaExit=6,
    AddRmsGws=7, AddRmsEnter=8, AddRmsExit=9, QmmRegistration=10
};
struct Event {
    uint32_t version{1}, size{sizeof(Event)}, kind{0}, flags{0};
    int32_t device{-1}, status{0};
    uint64_t sequence{0}, thread{0}, call{0}, registration{0}, executor{0}, cache{0};
    uint64_t producer{0}, output{0}, output_bytes{0}, residual{0}, residual_bytes{0};
    int64_t rows{0}, columns{0}, residual_rows{0}, residual_columns{0};
    int32_t dtype{-1}, format{-1};
    uint32_t dimensions{0}, residual_dimensions{0};
    int64_t shape[8]{}, strides[8]{}, residual_shape[8]{}, residual_strides[8]{};
};
struct State {
    uint32_t version{1}, size{sizeof(State)}, flags{0}, reserved{0};
    uint64_t capacity{0}, total_events{0}, stored_events{0}, overflow_events{0}, errors{0};
    uint64_t all_runs{0}, qmm_runs{0}, non_target_runs{0};
    uint64_t qmm_enter{0}, qmm_exit{0}, fia_before{0}, fia_gws{0}, fia_enter{0}, fia_exit{0};
    uint64_t add_rms_gws{0}, add_rms_enter{0}, add_rms_exit{0}, qmm_registration{0};
};
template <std::size_t Capacity> class Inventory {
    std::mutex mutex_;
    State state_{};
    std::array<Event,Capacity> events_{};
public:
    Inventory() { state_.capacity=Capacity; }
    void run(bool qmm) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++state_.all_runs;
        if(qmm) ++state_.qmm_runs; else ++state_.non_target_runs;
    }
    void error() { std::lock_guard<std::mutex> lock(mutex_);++state_.errors; }
    uint64_t append(Event event) {
        std::lock_guard<std::mutex> lock(mutex_);
        event.sequence=++state_.total_events;
        if(!event.call && (event.kind==QmmEnter || event.kind==FiaBefore))event.call=event.sequence;
        switch(event.kind) {
        case QmmEnter:++state_.qmm_enter;break;case QmmExit:++state_.qmm_exit;break;
        case FiaBefore:++state_.fia_before;break;case FiaGws:++state_.fia_gws;break;
        case FiaEnter:++state_.fia_enter;break;case FiaExit:++state_.fia_exit;break;
        case AddRmsGws:++state_.add_rms_gws;break;case AddRmsEnter:++state_.add_rms_enter;break;
        case AddRmsExit:++state_.add_rms_exit;break;case QmmRegistration:++state_.qmm_registration;break;
        default:++state_.errors;
        }
        if(state_.stored_events<Capacity) events_[state_.stored_events++]=event;
        else ++state_.overflow_events;
        return event.sequence;
    }
    State state() { std::lock_guard<std::mutex> lock(mutex_);return state_; }
    bool read(uint64_t index,Event* output) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!output || index>=state_.stored_events)return false;
        *output=events_[index];return true;
    }
};
} // namespace dcat26_inventory
#endif
