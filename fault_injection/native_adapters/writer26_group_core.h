#ifndef DCAT26_GROUP_CORE_V2_H
#define DCAT26_GROUP_CORE_V2_H
// Independent v2 serial transaction engine. Legacy writer/C1 controls are not changed.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace dcat26_group {
using Key=std::array<uint64_t,4>;
enum Mode : uint32_t { Reference=1, Lossless=2, Fault=3, Off=4 };
enum Mechanism : uint32_t { Qmm=1, Fia=2 };
enum Error : int32_t { Ok=0, Identity=-1, Busy=-2, Budget=-3, Geometry=-4, Qualification=-5,
    Allocation=-6, Copy=-7, Sync=-8, Writer=-9, Consumer=-10, Restore=-11, Free=-12, StateError=-13 };
enum class CopyKind { DeviceToHost, HostToDevice, DeviceToDevice };
struct Ops {
    virtual ~Ops()=default;
    virtual bool sync(uint64_t stream)=0;
    virtual bool allocate(void** output,size_t bytes)=0;
    virtual bool release(void* pointer)=0;
    virtual bool copy(void* output,size_t capacity,const void* input,size_t bytes,CopyKind kind)=0;
};
struct Config {
    uint32_t version{2},size{sizeof(Config)},mechanism{},max_targets{};
    uint64_t owner_hi{},owner_lo{},generation{},thread{},max_calls{},max_device_bytes{},max_host_bytes{};
};
struct Call {
    uint32_t version{2},size{sizeof(Call)},mechanism{},dimensions{};
    int32_t device{-1},dtype{-1},format{-1},reserved{};
    uint64_t token{},thread{},independent_call{},registration{},cache{},producer_sequence{};
    uint64_t output{},output_bytes{},residual{},residual_bytes{};
    int64_t shape[4]{};
};
struct Lease {
    uint32_t version{2},size{sizeof(Lease)},mode{},reserved{};
    uint64_t owner_hi{},owner_lo{},generation{},thread{};
    Key target{};
    Call expected{};
    uint64_t residual_allocation_end{},output_allocation_begin{},shift_bytes{};
};
struct Receipt {
    uint32_t version{2},size{sizeof(Receipt)},mode{},mechanism{};
    int32_t error{},writer_status{},consumer_status{},reserved{};
    uint64_t owner_hi{},owner_lo{},generation{},token{},independent_call{},registration{},cache{},producer_sequence{};
    uint64_t output{},output_bytes{},residual{},residual_bytes{},replacement{},stream{},executor{};
    Key target{};
    uint64_t writer_calls{},consumer_calls{},changed_residual_bytes{},writer_complete{},consumer_complete{};
    uint64_t output_readback_exact{},padding_readback_exact{},residual_unchanged{},residual_restored_exact{};
    uint64_t reference_equal{},lossless_qualified{},off_exact{},drained{},released{};
};
struct State {
    uint32_t version{2},size{sizeof(State)},mechanism{},reserved{};
    int32_t error{},reserved2{};
    uint64_t installed{},stopped{},sealed{},owner_hi{},owner_lo{},generation{},thread{};
    uint64_t targets{},calls{},pending{},executing{},device_bytes{},host_bytes{},descriptor_pending{};
};
inline bool valid_key(const Key& key) { return std::any_of(key.begin(),key.end(),[](uint64_t n){return n!=0;}); }
inline bool span(uint64_t address,uint64_t bytes) {
    return address && bytes && address<=std::numeric_limits<uint64_t>::max()-bytes;
}
inline bool overlap(uint64_t a,uint64_t an,uint64_t b,uint64_t bn) {
    return !span(a,an) || !span(b,bn) || (a<b+bn && b<a+an);
}
inline bool same_call(const Call& a,const Call& b) {
    return a.version==b.version && a.size==b.size && a.mechanism==b.mechanism && a.dimensions==b.dimensions
        && a.device==b.device && a.dtype==b.dtype && a.format==b.format && !a.reserved && !b.reserved
        && a.token==b.token && a.thread==b.thread && a.independent_call==b.independent_call
        && a.registration==b.registration && a.cache==b.cache && a.producer_sequence==b.producer_sequence
        && a.output==b.output && a.output_bytes==b.output_bytes && a.residual==b.residual
        && a.residual_bytes==b.residual_bytes && std::equal(a.shape,a.shape+4,b.shape);
}
inline bool same_site(const Call& a,const Call& b) {
    return a.mechanism==b.mechanism && a.dimensions==b.dimensions && a.device==b.device
        && a.dtype==b.dtype && a.format==b.format && a.output_bytes==b.output_bytes
        && a.residual_bytes==b.residual_bytes && std::equal(a.shape,a.shape+4,b.shape)
        && (a.mechanism!=Qmm || (a.cache==b.cache && a.registration==b.registration));
}
struct Target {
    Call site{};
    std::vector<unsigned char> output,residual;
    bool reference{},lossless{},off{};
    uint64_t faults{};
};
struct Transaction {
    bool active{},started{},executing{},drained{},writer_ok{},consumer_ok{},restored{};
    bool residual_backed{},output_backed{},padding_backed{},wrote{},descriptor_pending{};
    Call call{};Lease lease{};Receipt receipt{};
    uint64_t stream{},executor{},padding{},replacement{},device_bytes{};
    void* temporary{};
    std::vector<unsigned char> residual_before,output_before,padding_before,native_output;
    std::vector<unsigned char> residual_after,output_check,padding_check;
    uint64_t host_bytes() const {
        return residual_before.capacity()+output_before.capacity()+padding_before.capacity()+native_output.capacity()
            +residual_after.capacity()+output_check.capacity()+padding_check.capacity();
    }
};
class Engine {
    Ops& ops_;
    Config config_{};
    bool installed_{},stopped_{},sealed_{};
    int error_{};
    uint64_t calls_{},last_token_{},last_independent_{};
    std::map<Key,Target> targets_;
    std::map<std::pair<uint64_t,uint64_t>,uint64_t> generations_;
    Transaction tx_{};
    Receipt last_{};
    uint64_t references_bytes() const {
        uint64_t total=0;
        for(const auto& item:targets_)total+=item.second.output.capacity()+item.second.residual.capacity();
        return total;
    }
    bool owner(const Config& value) const {
        return installed_ && value.owner_hi==config_.owner_hi && value.owner_lo==config_.owner_lo
            && value.generation==config_.generation && value.thread==config_.thread;
    }
    int fail(int code) { error_=code;if(tx_.active)tx_.receipt.error=code;return code; }
    bool read(std::vector<unsigned char>& dst,uint64_t src,uint64_t bytes) {
        return ops_.copy(dst.data(),dst.size(),reinterpret_cast<void*>(src),bytes,CopyKind::DeviceToHost);
    }
    bool write(uint64_t dst,const std::vector<unsigned char>& src,uint64_t bytes) {
        return ops_.copy(reinterpret_cast<void*>(dst),bytes,src.data(),bytes,CopyKind::HostToDevice);
    }
    bool verify(uint64_t dst,const std::vector<unsigned char>& expected,std::vector<unsigned char>& scratch) {
        return read(scratch,dst,expected.size()) && scratch.size()==expected.size()
            && std::equal(expected.begin(),expected.end(),scratch.begin());
    }
    int restore_spans(bool abort_output) {
        if(!ops_.sync(tx_.stream))return fail(Sync);
        const auto& expected=(abort_output || tx_.native_output.empty() || !tx_.writer_ok)
            ?tx_.output_before:tx_.native_output;
        if(tx_.output_backed && (!write(tx_.call.output,expected,expected.size())
            || !verify(tx_.call.output,expected,tx_.output_check)))return fail(Restore);
        if(tx_.padding_backed && tx_.padding && (!write(tx_.call.residual+tx_.call.residual_bytes,tx_.padding_before,tx_.padding)
            || !verify(tx_.call.residual+tx_.call.residual_bytes,tx_.padding_before,tx_.padding_check)))return fail(Restore);
        if(tx_.residual_backed && (!write(tx_.call.residual,tx_.residual_before,tx_.call.residual_bytes)
            || !verify(tx_.call.residual,tx_.residual_before,tx_.residual_after)))return fail(Restore);
        if(!ops_.sync(tx_.stream))return fail(Sync);
        tx_.receipt.residual_restored_exact=tx_.residual_backed;
        tx_.receipt.padding_readback_exact=!tx_.padding || tx_.padding_backed;
        tx_.restored=true;return Ok;
    }
public:
    explicit Engine(Ops& ops):ops_(ops){}
    bool matches_owner(uint64_t hi,uint64_t lo,uint64_t generation,uint64_t thread) const {
        Config value{};value.owner_hi=hi;value.owner_lo=lo;value.generation=generation;value.thread=thread;
        return owner(value);
    }
    int open(const Config& value) {
        const auto key=std::make_pair(value.owner_hi,value.owner_lo);
        if(installed_ || tx_.active || value.version!=2 || value.size!=sizeof(Config)
            || (!value.owner_hi && !value.owner_lo) || !value.generation || !value.thread
            || (value.mechanism!=Qmm && value.mechanism!=Fia) || !value.max_targets || value.max_targets>512
            || !value.max_calls || value.max_calls>8192 || value.max_device_bytes>512ULL*1024*1024
            || !value.max_host_bytes || value.max_host_bytes>2ULL*1024*1024*1024
            || generations_.size()>=64 || value.generation<=generations_[key])return Identity;
        config_=value;generations_[key]=value.generation;installed_=true;stopped_=sealed_=false;
        calls_=last_token_=last_independent_=0;error_=0;targets_.clear();last_=Receipt{};return Ok;
    }
    int prepare(const Call& call,const Lease& lease) {
        if(!matches_owner(lease.owner_hi,lease.owner_lo,lease.generation,lease.thread)
            || lease.version!=2 || lease.size!=sizeof(Lease) || lease.reserved || !same_call(call,lease.expected)
            || call.version!=2 || call.size!=sizeof(Call) || call.mechanism!=config_.mechanism
            || call.thread!=config_.thread || call.token<=last_token_ || call.independent_call<=last_independent_
            || !valid_key(lease.target))return Identity;
        if(stopped_ || tx_.active)return Busy;
        if(error_ || calls_>=config_.max_calls)return Budget;
        if(lease.mode<Reference || lease.mode>Off || call.device<0 || !call.producer_sequence
            || (call.dtype!=1 && call.dtype!=27) || call.format!=2 || !call.dimensions || call.dimensions>4
            || !span(call.output,call.output_bytes) || !span(call.residual,call.residual_bytes)
            || call.output_bytes>8*1024*1024 || call.residual_bytes>8*1024*1024
            || overlap(call.output,call.output_bytes,call.residual,call.residual_bytes))return Geometry;
        uint64_t elements=1;
        for(uint32_t i=0;i<call.dimensions;++i) {
            if(call.shape[i]<=0 || static_cast<uint64_t>(call.shape[i])>8*1024*1024/elements)return Geometry;
            elements*=static_cast<uint64_t>(call.shape[i]);
        }
        if(elements*2!=call.output_bytes)return Geometry;
        uint64_t padding=0;
        if(call.mechanism==Qmm) {
            if(!call.cache || !call.registration || lease.output_allocation_begin!=call.output
                || lease.residual_allocation_end!=call.output || call.residual+call.residual_bytes>call.output)return Geometry;
            padding=call.output-(call.residual+call.residual_bytes);
            if(padding>4064 || lease.shift_bytes!=padding+32 || call.residual_bytes<32
                || call.output_bytes<lease.shift_bytes)return Geometry;
        } else if(call.output_bytes>call.residual_bytes || lease.shift_bytes || lease.output_allocation_begin
            || lease.residual_allocation_end)return Geometry;
        auto target=targets_.find(lease.target);
        if(target==targets_.end()) {
            if(sealed_ || lease.mode!=Reference || targets_.size()>=config_.max_targets)return Qualification;
            target=targets_.emplace(lease.target,Target{}).first;target->second.site=call;
        }
        if(!same_site(target->second.site,call))return Identity;
        if(lease.mode==Reference && (sealed_ || target->second.reference))return Qualification;
        if(lease.mode!=Reference && !target->second.reference)return Qualification;
        if(lease.mode==Fault && (!sealed_ || !target->second.lossless || !target->second.off || target->second.faults>=8))return Qualification;
        if(lease.mode==Lossless && sealed_)return Qualification;
        const uint64_t need=call.residual_bytes*2+call.output_bytes*3+padding*2;
        const uint64_t extra=lease.mode==Reference?call.output_bytes+call.residual_bytes:0;
        if(need+extra+references_bytes()>config_.max_host_bytes
            || (lease.mode==Lossless && call.output_bytes>config_.max_device_bytes))return Budget;
        tx_=Transaction{};tx_.active=true;tx_.call=call;tx_.lease=lease;tx_.padding=padding;
        tx_.receipt.mode=lease.mode;tx_.receipt.mechanism=call.mechanism;tx_.receipt.owner_hi=lease.owner_hi;
        tx_.receipt.owner_lo=lease.owner_lo;tx_.receipt.generation=lease.generation;tx_.receipt.target=lease.target;
        tx_.receipt.token=call.token;tx_.receipt.independent_call=call.independent_call;
        tx_.receipt.registration=call.registration;tx_.receipt.cache=call.cache;tx_.receipt.producer_sequence=call.producer_sequence;
        tx_.receipt.output=call.output;tx_.receipt.output_bytes=call.output_bytes;
        tx_.receipt.residual=call.residual;tx_.receipt.residual_bytes=call.residual_bytes;
        ++calls_;last_token_=call.token;last_independent_=call.independent_call;
        try {
            tx_.residual_before.resize(call.residual_bytes);tx_.residual_after.resize(call.residual_bytes);
            tx_.output_before.resize(call.output_bytes);tx_.native_output.resize(call.output_bytes);tx_.output_check.resize(call.output_bytes);
            tx_.padding_before.resize(padding);tx_.padding_check.resize(padding);
        } catch(...) {return fail(Allocation);}
        if(tx_.host_bytes()+extra+references_bytes()>config_.max_host_bytes)return fail(Budget);
        tx_.replacement=call.output;
        if(lease.mode==Lossless) {
            if(!ops_.allocate(&tx_.temporary,call.output_bytes) || !tx_.temporary)return fail(Allocation);
            tx_.device_bytes=call.output_bytes;tx_.replacement=reinterpret_cast<uint64_t>(tx_.temporary);
            if(overlap(tx_.replacement,call.output_bytes,call.output,call.output_bytes)
                || overlap(tx_.replacement,call.output_bytes,call.residual,call.residual_bytes))return fail(Allocation);
        } else if(lease.mode==Fault) {
            ++target->second.faults; // A failed native attempt still spends the per-target budget.
            tx_.replacement=call.mechanism==Qmm?call.output-lease.shift_bytes:call.residual;
        }
        tx_.receipt.replacement=tx_.replacement;return Ok;
    }
    uint64_t replacement() const {return tx_.active?tx_.replacement:0;}
    uint64_t token() const {return tx_.active?tx_.call.token:0;}
    uint32_t mechanism() const {return config_.mechanism;}
    int reject(int code) {return fail(code==Ok?Identity:code);}
    int descriptor(bool pending) {if(!tx_.active)return StateError;tx_.descriptor_pending=pending;return Ok;}
    int start(uint64_t token,uint64_t stream,uint64_t executor=0,uint64_t registration=0) {
        if(!tx_.active || token!=tx_.call.token || tx_.started || error_ || !stream
            || (tx_.call.mechanism==Fia && (!executor || !registration)))return StateError;
        tx_.stream=stream;tx_.executor=executor;tx_.receipt.stream=stream;tx_.receipt.executor=executor;
        if(tx_.call.mechanism==Fia)tx_.receipt.registration=registration; // Actual GWS-to-Execute lifetime only.
        if(!ops_.sync(stream))return fail(Sync);
        if(!read(tx_.residual_before,tx_.call.residual,tx_.call.residual_bytes))return fail(Copy);
        tx_.residual_backed=true;
        if(!read(tx_.output_before,tx_.call.output,tx_.call.output_bytes))return fail(Copy);
        tx_.output_backed=true;
        if(tx_.padding && !read(tx_.padding_before,tx_.call.residual+tx_.call.residual_bytes,tx_.padding))return fail(Copy);
        tx_.padding_backed=true;
        if(tx_.lease.mode!=Fault && tx_.lease.mode!=Reference
            && tx_.residual_before!=targets_.at(tx_.lease.target).residual)return fail(Qualification);
        tx_.started=true;tx_.executing=true;return Ok;
    }
    int writer_done(uint64_t token,uint64_t stream,int status,uint64_t original_calls) {
        if(!tx_.active || token!=tx_.call.token || !tx_.started || !tx_.executing || stream!=tx_.stream)return StateError;
        tx_.executing=false;tx_.wrote=true;tx_.receipt.writer_calls=original_calls;tx_.receipt.writer_status=status;
        if(status || original_calls!=1)return fail(Writer);
        if(!ops_.sync(stream))return fail(Sync);
        if(!read(tx_.native_output,tx_.replacement,tx_.call.output_bytes)
            || !read(tx_.residual_after,tx_.call.residual,tx_.call.residual_bytes))return fail(Copy);
        tx_.receipt.changed_residual_bytes=0;
        for(size_t i=0;i<tx_.residual_before.size();++i)
            tx_.receipt.changed_residual_bytes+=tx_.residual_before[i]!=tx_.residual_after[i];
        tx_.receipt.residual_unchanged=tx_.residual_before==tx_.residual_after;
        if(tx_.lease.mode==Fault) {
            if(!tx_.receipt.changed_residual_bytes)return fail(Writer);
            const size_t begin=tx_.call.mechanism==Qmm?tx_.call.residual_bytes-32:0;
            const size_t end=tx_.call.mechanism==Qmm?tx_.call.residual_bytes:tx_.call.output_bytes;
            for(size_t i=0;i<tx_.residual_before.size();++i)
                if((i<begin || i>=end) && tx_.residual_before[i]!=tx_.residual_after[i])return fail(Writer);
        } else if(!tx_.receipt.residual_unchanged)return fail(Writer);
        if(tx_.replacement!=tx_.call.output) {
            // The QMM shifted source overlaps the original output. Preserve the
            // legacy host-staged copy; overlapping device memcpy is not valid.
            const bool copied=tx_.call.mechanism==Qmm && tx_.lease.mode==Fault
                ?write(tx_.call.output,tx_.native_output,tx_.call.output_bytes)
                :ops_.copy(reinterpret_cast<void*>(tx_.call.output),tx_.call.output_bytes,
                    reinterpret_cast<void*>(tx_.replacement),tx_.call.output_bytes,CopyKind::DeviceToDevice);
            if(!copied)return fail(Copy);
        }
        if(!verify(tx_.call.output,tx_.native_output,tx_.output_check))return fail(Copy);
        tx_.receipt.output_readback_exact=1;
        if(tx_.padding && (!write(tx_.call.residual+tx_.call.residual_bytes,tx_.padding_before,tx_.padding)
            || !verify(tx_.call.residual+tx_.call.residual_bytes,tx_.padding_before,tx_.padding_check)))return fail(Restore);
        tx_.receipt.padding_readback_exact=1;
        auto& target=targets_.at(tx_.lease.target);
        if(tx_.lease.mode!=Fault && tx_.lease.mode!=Reference) {
            tx_.receipt.reference_equal=target.output==tx_.native_output && target.residual==tx_.residual_before;
            if(!tx_.receipt.reference_equal)return fail(Qualification);
        }
        tx_.writer_ok=true;tx_.receipt.writer_complete=1;return Ok;
    }
    int failed_writer_completion(uint64_t token,uint64_t stream,int status,uint64_t original_calls,int reason) {
        // The exact callback token/stream proves that the original returned,
        // even when independent completion metadata failed validation. Keep
        // every backup; only an explicit stopped/drained OFF may restore it.
        if(!tx_.active || token!=tx_.call.token || !tx_.started || !tx_.executing || stream!=tx_.stream)return StateError;
        tx_.executing=false;tx_.wrote=true;tx_.receipt.writer_calls=original_calls;tx_.receipt.writer_status=status;
        return fail(reason);
    }
    int consumer_done(uint64_t producer,uint64_t residual,uint64_t residual_bytes,uint64_t output,
        uint64_t output_bytes,int device,uint64_t stream,int status,uint64_t original_calls) {
        if(!tx_.active || !tx_.writer_ok || tx_.consumer_ok)return StateError;
        tx_.receipt.consumer_status=status;tx_.receipt.consumer_calls=original_calls;
        if(producer!=tx_.call.producer_sequence+1 || residual!=tx_.call.residual || residual_bytes!=tx_.call.residual_bytes
            || device!=tx_.call.device || stream!=tx_.stream || overlap(residual,residual_bytes,output,output_bytes)
            || status || original_calls!=1)return fail(Consumer);
        tx_.consumer_ok=true;tx_.receipt.consumer_complete=1;
        if(restore_spans(false)!=Ok)return error_;
        auto& target=targets_.at(tx_.lease.target);
        if(tx_.lease.mode==Reference) {
            try {target.output=tx_.native_output;target.residual=tx_.residual_before;}
            catch(...) {return fail(Allocation);}
            target.reference=true;tx_.receipt.reference_equal=1;
        } else if(tx_.lease.mode==Lossless)target.lossless=true;
        else if(tx_.lease.mode==Off)target.off=true;
        tx_.receipt.lossless_qualified=target.lossless;
        tx_.receipt.off_exact=tx_.lease.mode==Off && target.off;
        return Ok;
    }
    int seal(const Config& owner_value,const std::vector<Key>& keys) {
        if(!owner(owner_value) || stopped_ || error_ || tx_.active || sealed_ || keys.empty() || keys.size()!=targets_.size())return StateError;
        std::map<Key,bool> unique;
        for(const auto& key:keys) {
            const auto it=targets_.find(key);
            if(!unique.emplace(key,true).second || it==targets_.end() || !it->second.reference || !it->second.lossless || !it->second.off)return Qualification;
        }
        sealed_=true;return Ok;
    }
    int acknowledge(const Config& value,uint64_t token) {
        if(!owner(value) || !tx_.active || tx_.call.token!=token || tx_.executing || tx_.descriptor_pending)return StateError;
        if(!tx_.restored || (!stopped_ && (error_ || !tx_.consumer_ok)))return StateError;
        if(tx_.temporary && !ops_.release(tx_.temporary))return fail(Free);
        tx_.temporary=nullptr;tx_.device_bytes=0;tx_.receipt.released=1;
        last_=tx_.receipt;tx_=Transaction{};return Ok;
    }
    int stop(const Config& value) {if(!owner(value))return Identity;stopped_=true;return Ok;}
    int drain(const Config& value) {
        if(!owner(value) || !stopped_ || tx_.executing)return StateError;
        if(tx_.active && tx_.stream && !ops_.sync(tx_.stream))return fail(Sync);
        if(tx_.active){tx_.drained=true;tx_.receipt.drained=1;}return Ok;
    }
    int restore(const Config& value,uint64_t token) {
        if(!owner(value) || !stopped_ || !tx_.active || tx_.call.token!=token || !tx_.drained || tx_.executing)return StateError;
        if(!tx_.wrote){tx_.restored=true;return Ok;} // No device write occurred; backups may be incomplete.
        return restore_spans(!tx_.writer_ok);
    }
    int close(const Config& value) {
        if(!owner(value) || !stopped_ || tx_.active)return StateError;
        targets_.clear();installed_=false;return Ok;
    }
    Receipt receipt(uint64_t token) const {
        if(tx_.active && tx_.call.token==token)return tx_.receipt;
        if(last_.token==token)return last_;
        Receipt missing{};missing.error=Identity;return missing;
    }
    State state() const {
        State value{};value.mechanism=config_.mechanism;value.error=error_;value.installed=installed_;value.stopped=stopped_;
        value.sealed=sealed_;value.owner_hi=config_.owner_hi;value.owner_lo=config_.owner_lo;value.generation=config_.generation;
        value.thread=config_.thread;value.targets=targets_.size();value.calls=calls_;value.pending=tx_.active;
        value.executing=tx_.executing;value.device_bytes=tx_.device_bytes;value.host_bytes=references_bytes()+tx_.host_bytes();
        value.descriptor_pending=tx_.descriptor_pending;return value;
    }
};
}
#endif
