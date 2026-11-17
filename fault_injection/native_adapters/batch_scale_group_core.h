#ifndef DCAT_DQ_GROUP_CORE_V1_H
#define DCAT_DQ_GROUP_CORE_V1_H
// Independent group variant. No legacy C1 control or arithmetic is replaced.
// Ops is a device/descriptor boundary; host tests supply explicit memory mocks.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace dcat_dq_group {
using Key=std::array<uint64_t,4>;
enum Mode : uint32_t { Reference=1, Lossless=2, Fault=3, Off=4 };
enum Role : uint32_t { CorrectReference=1, ModelEffect=2 };
enum Kind : uint32_t { PrepareEnter=1, PrepareExit=2, ExecuteEnter=3, ExecuteExit=4 };
enum Error : int32_t { Ok=0, Identity=-1, Busy=-2, Budget=-3, Geometry=-4, Qualification=-5,
    Allocation=-6, Copy=-7, Sync=-8, Native=-9, Pending=-10, InputChanged=-11, NoEffect=-12,
    Restore=-13, Free=-14, Observer=-15, StateError=-16, Liveness=-17 };
enum class CopyKind { DeviceToHost, HostToDevice, DeviceToDevice };
struct Ops {
    virtual ~Ops()=default;
    virtual bool allocate(void** output,size_t bytes)=0;
    virtual bool release(void* pointer)=0;
    virtual bool copy(void* output,size_t capacity,const void* input,size_t bytes,CopyKind kind)=0;
    virtual bool sync(uint64_t stream)=0;
    virtual bool destroy_descriptor(uint64_t descriptor)=0;
};
struct View {
    uint64_t address{},bytes{},descriptor{};
    int64_t rows{},columns{},stride0{},stride1{},offset{};
    int32_t dtype{-1},format{-1};
    uint32_t dimensions{},reserved{};
};
struct Call {
    uint32_t version{1},size{sizeof(Call)};
    uint64_t token{},thread{},lifetime_token{};
    int32_t device{-1},reserved{};
    View input{},quant{},scale{};
};
struct Config {
    uint32_t version{1},size{sizeof(Config)},max_targets{},max_calls{};
    uint64_t owner_hi{},owner_lo{},generation{},thread{},max_device_bytes{},max_host_bytes{};
    int32_t device{-1},reserved{};
};
struct Lease {
    uint32_t version{1},size{sizeof(Lease)},mode{},reserved{};
    uint64_t owner_hi{},owner_lo{},generation{},thread{};
    Key target{},actual_row_proof{};
    Call expected{};
};
struct NativeEvent {
    uint32_t kind{},reserved{};
    uint64_t sequence{},registration{},thread{},executor{},stream{};
    int32_t device{-1},status{};
    uint64_t original_calls{};
    uint64_t workspace_required{},workspace{},workspace_capacity{};
    uint64_t smooth_tensor{},group_index_tensor{},offset_tensor{};
    int64_t destination_dtype{2};
    View input{},quant{},scale{};
};
struct ChildBinding {
    uint64_t parent_token{},child_lease{},owner_hi{},owner_lo{},generation{};
    uint32_t role{},reserved{};
};
struct JournalState {
    uint64_t total{},capacity{},overflow{},errors{},model_parents{},native_prepares{},native_executes{},
        helper_prepares{},model_prepares{},unclassified_prepares{},preparing{},pending{},executing{};
};
inline bool key_valid(const Key& key) { return std::any_of(key.begin(),key.end(),[](uint64_t n){return n!=0;}); }
inline bool span(uint64_t address,uint64_t bytes) {
    return address && bytes && address<=std::numeric_limits<uint64_t>::max()-bytes;
}
inline bool overlap(const View& a,const View& b) {
    return !span(a.address,a.bytes) || !span(b.address,b.bytes)
        || (a.address<b.address+b.bytes && b.address<a.address+a.bytes);
}
inline bool same_view(const View& a,const View& b,bool address=true,bool descriptor=true) {
    return (!address || a.address==b.address) && (!descriptor || a.descriptor==b.descriptor)
        && a.bytes==b.bytes && a.rows==b.rows && a.columns==b.columns && a.stride0==b.stride0
        && a.stride1==b.stride1 && a.offset==b.offset && a.dtype==b.dtype && a.format==b.format
        && a.dimensions==b.dimensions && !a.reserved && !b.reserved;
}
inline bool same_call(const Call& a,const Call& b) {
    return a.version==b.version && a.size==b.size && a.token==b.token && a.thread==b.thread && a.lifetime_token==b.lifetime_token
        && a.device==b.device && !a.reserved && !b.reserved
        && same_view(a.input,b.input) && same_view(a.quant,b.quant) && same_view(a.scale,b.scale);
}
inline bool same_site(const Call& a,const Call& b) {
    return a.device==b.device && same_view(a.input,b.input,false,false)
        && same_view(a.quant,b.quant,false,false) && same_view(a.scale,b.scale,false,false);
}
inline bool valid_call(const Call& c) {
    const auto& x=c.input;const auto& q=c.quant;const auto& s=c.scale;
    if(c.version!=1 || c.size!=sizeof(Call) || !c.token || !c.thread || !c.lifetime_token || c.device<0 || c.reserved
       || x.rows<1 || x.rows>64 || x.columns<1 || x.columns>65536 || x.rows*x.columns>1048576)return false;
    return (x.dtype==1 || x.dtype==27) && x.dimensions==2 && q.dimensions==2 && s.dimensions==1
        && x.format==2 && q.format==2 && s.format==2 && q.dtype==2 && s.dtype==0
        && x.offset==0 && q.offset==0 && s.offset==0 && !x.reserved && !q.reserved && !s.reserved
        && x.stride0==x.columns && x.stride1==1 && q.rows==x.rows && q.columns==x.columns
        && q.stride0==q.columns && q.stride1==1 && s.rows==x.rows && s.columns==0 && s.stride0==1 && s.stride1==0
        && x.bytes==uint64_t(x.rows*x.columns)*2 && q.bytes==uint64_t(x.rows*x.columns) && s.bytes==uint64_t(x.rows)*4
        && x.descriptor && q.descriptor && s.descriptor && !overlap(x,q) && !overlap(x,s) && !overlap(q,s);
}

// This census has no owner/selection logic. Actual shim call sites must invoke
// it BEFORE classification/admission and BEFORE each real dlsym GWS/execute.
class Journal {
    uint64_t capacity_,sequence_{};JournalState state_{};
    std::map<uint64_t,Call> parents_;
    std::map<uint64_t,NativeEvent> events_;
    std::map<uint64_t,uint64_t> pending_;
    std::map<uint64_t,uint64_t> prepared_exits_;
    std::set<uint64_t> preparing_;
    std::set<uint64_t> executing_;
    std::map<uint64_t,ChildBinding> bindings_;
    uint64_t append(NativeEvent event) {
        event.sequence=++sequence_;++state_.total;
        if(state_.total>capacity_){++state_.overflow;return 0;}
        events_.emplace(event.sequence,event);return event.sequence;
    }
public:
    explicit Journal(uint64_t capacity=32768):capacity_(capacity){state_.capacity=capacity;}
    Call parent(Call value) {
        value.token=++sequence_;++state_.total;++state_.model_parents;
        if(state_.total>capacity_){++state_.overflow;value.token=0;return value;}
        parents_.emplace(value.token,value);return value;
    }
    const Call* parent_record(uint64_t token) const {
        auto p=parents_.find(token);return p==parents_.end()?nullptr:&p->second;
    }
    const NativeEvent* event(uint64_t ticket) const {
        auto p=events_.find(ticket);return p==events_.end()?nullptr:&p->second;
    }
    uint64_t prepare(const View& x,const View& q,const View& s,uint64_t thread,int device,
        uint64_t smooth=0,uint64_t group_index=0,uint64_t offset=0,int64_t destination_dtype=2) {
        NativeEvent e{};e.kind=PrepareEnter;e.input=x;e.quant=q;e.scale=s;e.thread=thread;e.device=device;
        e.smooth_tensor=smooth;e.group_index_tensor=group_index;e.offset_tensor=offset;e.destination_dtype=destination_dtype;
        e.registration=sequence_+1;++state_.native_prepares;
        const auto ticket=append(e);if(ticket)preparing_.insert(ticket);return ticket;
    }
    uint64_t prepared(uint64_t begin,uint64_t executor,int status,uint64_t original_calls,uint64_t workspace_required=0) {
        const auto* source=event(begin);NativeEvent e{};
        if(!source || source->kind!=PrepareEnter){++state_.errors;}else e=*source;
        e.kind=PrepareExit;e.executor=executor;e.status=status;e.original_calls=original_calls;e.workspace_required=workspace_required;
        if(!status && !executor)++state_.errors;
        const auto ticket=append(e);
        if(source && source->kind==PrepareEnter && ticket) {
            if(!preparing_.erase(begin))++state_.errors;
            if(!prepared_exits_.emplace(begin,ticket).second)++state_.errors;
            if(executor && !pending_.emplace(executor,begin).second)++state_.errors;
        }
        return ticket;
    }
    uint64_t execute(uint64_t executor,uint64_t thread,int device,uint64_t stream,uint64_t workspace=0,uint64_t capacity=0) {
        NativeEvent e{};const auto found=pending_.find(executor);
        if(found==pending_.end() || !executing_.insert(executor).second){++state_.errors;}
        else {
            const auto exit=prepared_exits_.find(found->second);
            const auto* source=exit==prepared_exits_.end()?nullptr:event(exit->second);
            if(source)e=*source;else ++state_.errors;
        }
        e.kind=ExecuteEnter;e.executor=executor;e.thread=thread;e.device=device;e.stream=stream;e.original_calls=0;
        e.workspace=workspace;e.workspace_capacity=capacity;
        ++state_.native_executes;return append(e);
    }
    uint64_t executed(uint64_t begin,int status,uint64_t original_calls) {
        const auto* source=event(begin);NativeEvent e{};
        if(!source || source->kind!=ExecuteEnter){++state_.errors;}else e=*source;
        e.kind=ExecuteExit;e.status=status;e.original_calls=original_calls;
        // A nonzero real execute result has no established executor-consumption
        // contract. Stream completion alone cannot release that registration.
        if(original_calls==1 && status==0) {if(!pending_.erase(e.executor))++state_.errors;}
        if(!executing_.erase(e.executor))++state_.errors;
        return append(e);
    }
    bool classify(uint64_t prepare,const ChildBinding& binding) {
        const auto* e=event(prepare);
        if(!e || e->kind!=PrepareEnter || prepare<=binding.parent_token || !parent_record(binding.parent_token)
           || !binding.child_lease || !binding.owner_hi || !binding.owner_lo || !binding.generation
           || (binding.role!=CorrectReference && binding.role!=ModelEffect) || binding.reserved
           || !bindings_.emplace(prepare,binding).second)return false;
        return true;
    }
    bool unaccounted_references_since(uint64_t parent,const std::vector<View>& spans,const std::set<uint64_t>& callbacks) const {
        for(const auto& item:events_) {
            const auto& e=item.second;if(e.sequence<=parent)continue;
            const auto binding=bindings_.find(e.registration);
            if(binding!=bindings_.end() && binding->second.parent_token==parent && callbacks.count(e.sequence))continue;
            for(const auto& watched:spans) {
                if(overlap(e.input,watched) || overlap(e.quant,watched) || overlap(e.scale,watched))return true;
                if(e.workspace_capacity){View w{};w.address=e.workspace;w.bytes=e.workspace_capacity;if(overlap(w,watched))return true;}
            }
        }
        return false;
    }
    JournalState state() const {
        auto s=state_;s.preparing=preparing_.size();s.pending=pending_.size();s.executing=executing_.size();
        for(const auto& b:bindings_)(b.second.role==CorrectReference?s.helper_prepares:s.model_prepares)++;
        s.unclassified_prepares=s.native_prepares-s.helper_prepares-s.model_prepares;return s;
    }
    const ChildBinding* binding(uint64_t registration) const {
        const auto found=bindings_.find(registration);return found==bindings_.end()?nullptr:&found->second;
    }
    uint64_t last_sequence() const {return sequence_;}
    void mark_error() noexcept {++state_.errors;}
};

struct ChildPlan {
    uint32_t role{},reserved{};
    uint64_t child_lease{},parent_token{};
    View input{},quant{},scale{};
};
struct ChildReceipt {
    ChildPlan plan{};
    uint64_t prepare_enter{},prepare_exit{},execute_enter{},execute_exit{},registration{},executor{},stream{};
    uint64_t prepare_original_calls{},execute_original_calls{},prepared{},returned{},drained{},readback{};
    uint64_t workspace_required{},workspace{},workspace_capacity{},workspace_group_owned{};
    int32_t prepare_status{},execute_status{};
};
struct Receipt {
    uint32_t version{1},size{sizeof(Receipt)},mode{},reserved{};
    int32_t error{},last_error{};
    uint64_t owner_hi{},owner_lo{},generation{},parent_token{},lifetime_token{};
    Key target{},actual_row_proof{};
    ChildReceipt reference{},effect{};
    uint64_t input_before_captured{},input_unchanged{},reference_effect_compared{},changed_quant_bytes{},changed_scale_bytes{},
        published{},publication_exact{},public_bytes_restored{},released{},unsafe_allocation_address{},unsafe_allocation_bytes{};
};
struct State {
    uint64_t installed{},stopped{},sealed{},owner_hi{},owner_lo{},generation{},targets{},admitted_model_calls{},
        pending_parent{},native_child_prepares{},native_child_executes{},device_bytes{},host_bytes{},owned_descriptors{},
        stopped_targets{},drained_targets{},owned_workspace_bytes{},borrowed_workspace_bytes{},quarantined_bytes{};
    int32_t error{};
};
struct LiveWitness {
    // The shim must mint this token while strongly retaining the actual public
    // tensors/allocator lifetime. Equal pointer/shape is not a live witness.
    uint64_t parent_token{},lifetime_token{};View input{},quant{},scale{};
};
struct Target {
    Call site{};std::vector<unsigned char> input,quant,scale;
    bool reference{},lossless{},off{},stopped{},drained{};uint64_t last_stream{},faults{};
};
struct Transaction {
    bool active{},started{},published{},restored{};
    Call call{};Lease lease{};Receipt receipt{};uint64_t stream{};
    std::array<void*,4> buffers{};std::array<uint64_t,4> buffer_bytes{};
    void* helper_workspace{};uint64_t helper_workspace_bytes{},borrowed_workspace{},borrowed_workspace_bytes{};
    std::vector<uint64_t> descriptors;
    std::vector<unsigned char> input_before,public_quant_before,public_scale_before,input_check,
        reference_quant,reference_scale,effect_quant,effect_scale,scratch;
    uint64_t host_bytes() const {
        return input_before.capacity()+public_quant_before.capacity()+public_scale_before.capacity()+input_check.capacity()
            +reference_quant.capacity()+reference_scale.capacity()+effect_quant.capacity()+effect_scale.capacity()+scratch.capacity();
    }
};
class Engine {
    Ops& ops_;Journal& journal_;Config config_{};bool installed_{},stopped_{},sealed_{};
    int error_{};uint64_t admitted_{},child_prepares_{},child_executes_{},last_parent_{},opened_sequence_{};
    std::map<Key,Target> targets_;std::map<std::pair<uint64_t,uint64_t>,uint64_t> generations_;
    Transaction tx_{};Receipt last_{};
    bool owner(const Config& c) const {
        return installed_ && c.version==1 && c.size==sizeof(Config) && !c.reserved
            && c.owner_hi==config_.owner_hi && c.owner_lo==config_.owner_lo
            && c.generation==config_.generation && c.thread==config_.thread && c.device==config_.device;
    }
    int fail(int error) {
        error_=error;
        if(tx_.active){if(!tx_.receipt.error)tx_.receipt.error=error;tx_.receipt.last_error=error;}
        return error;
    }
    uint64_t target_bytes() const {
        uint64_t n=0;for(const auto& v:targets_)n+=v.second.input.capacity()+v.second.quant.capacity()+v.second.scale.capacity();return n;
    }
    ChildReceipt* child(uint32_t role) {
        if(role==CorrectReference && tx_.lease.mode==Fault)return &tx_.receipt.reference;
        if(role==ModelEffect)return &tx_.receipt.effect;
        return nullptr;
    }
    bool read(std::vector<unsigned char>& out,uint64_t pointer,size_t bytes) {
        out.resize(bytes);return ops_.copy(out.data(),out.size(),reinterpret_cast<void*>(pointer),bytes,CopyKind::DeviceToHost);
    }
    bool unchanged() {
        tx_.receipt.input_unchanged=0;
        if(!read(tx_.input_check,tx_.call.input.address,tx_.call.input.bytes))return false;
        tx_.receipt.input_unchanged=tx_.input_before==tx_.input_check;
        return tx_.receipt.input_unchanged!=0;
    }
    bool write_exact(uint64_t pointer,const std::vector<unsigned char>& value) {
        if(!ops_.copy(reinterpret_cast<void*>(pointer),value.size(),value.data(),value.size(),CopyKind::HostToDevice))return false;
        return read(tx_.scratch,pointer,value.size()) && tx_.scratch==value;
    }
    static uint64_t differences(const std::vector<unsigned char>& a,const std::vector<unsigned char>& b) {
        if(a.size()!=b.size())return std::numeric_limits<uint64_t>::max();
        uint64_t n=0;for(size_t i=0;i<a.size();++i)n+=a[i]!=b[i];return n;
    }
    bool disjoint_from_resources(uint64_t address,uint64_t bytes) const {
        View candidate{};candidate.address=address;candidate.bytes=bytes;
        if(!span(address,bytes) || overlap(candidate,tx_.call.input) || overlap(candidate,tx_.call.quant)
           || overlap(candidate,tx_.call.scale))return false;
        for(size_t i=0;i<tx_.buffers.size();++i)if(tx_.buffers[i]) {
            View v{};v.address=reinterpret_cast<uint64_t>(tx_.buffers[i]);v.bytes=tx_.buffer_bytes[i];
            if(overlap(candidate,v))return false;
        }
        if(tx_.helper_workspace) {
            View v{};v.address=reinterpret_cast<uint64_t>(tx_.helper_workspace);v.bytes=tx_.helper_workspace_bytes;
            if(overlap(candidate,v))return false;
        }
        return true;
    }
    bool allocate_owned(void*& pointer,uint64_t& capacity,uint64_t bytes) {
        void* actual=nullptr;const bool ok=ops_.allocate(&actual,bytes);
        if(!actual)return false;
        if(!disjoint_from_resources(reinterpret_cast<uint64_t>(actual),bytes)) {
            // Ownership cannot be inferred for an alias of public/other owned
            // memory. Never free this pointer as a supposedly new allocation.
            tx_.receipt.unsafe_allocation_address=reinterpret_cast<uint64_t>(actual);
            tx_.receipt.unsafe_allocation_bytes=bytes;return false;
        }
        pointer=actual;capacity=bytes;return ok;
    }
    bool allocate(size_t index,uint64_t bytes) {
        return allocate_owned(tx_.buffers[index],tx_.buffer_bytes[index],bytes);
    }
    int release_owned() {
        for(auto& descriptor:tx_.descriptors) {
            if(descriptor && !ops_.destroy_descriptor(descriptor))return fail(Free);
            descriptor=0;
        }
        for(size_t i=0;i<tx_.buffers.size();++i) {
            if(tx_.buffers[i] && !ops_.release(tx_.buffers[i]))return fail(Free);
            tx_.buffers[i]=nullptr;tx_.buffer_bytes[i]=0;
        }
        if(tx_.helper_workspace && !ops_.release(tx_.helper_workspace))return fail(Free);
        tx_.helper_workspace=nullptr;tx_.helper_workspace_bytes=0;
        return Ok;
    }
    uint64_t owned_device_bytes() const {
        uint64_t n=tx_.helper_workspace_bytes;for(auto v:tx_.buffer_bytes)n+=v;return n;
    }
    bool raw_quiescent() const {
        const auto state=journal_.state();
        if(state.preparing || state.pending || state.executing || state.errors || state.overflow)return false;
        if(!tx_.active)return true;
        if(tx_.receipt.unsafe_allocation_bytes)return false;
        std::vector<View> spans{tx_.call.input,tx_.call.quant,tx_.call.scale};
        for(size_t i=0;i<tx_.buffers.size();++i)if(tx_.buffers[i]) {
            View v{};v.address=reinterpret_cast<uint64_t>(tx_.buffers[i]);v.bytes=tx_.buffer_bytes[i];spans.push_back(v);
        }
        if(tx_.helper_workspace){View v{};v.address=reinterpret_cast<uint64_t>(tx_.helper_workspace);v.bytes=tx_.helper_workspace_bytes;spans.push_back(v);}
        if(tx_.borrowed_workspace){View v{};v.address=tx_.borrowed_workspace;v.bytes=tx_.borrowed_workspace_bytes;spans.push_back(v);}
        const auto& r=tx_.receipt.reference;const auto& e=tx_.receipt.effect;
        const std::set<uint64_t> callbacks{r.prepare_enter,r.prepare_exit,r.execute_enter,r.execute_exit,
            e.prepare_enter,e.prepare_exit,e.execute_enter,e.execute_exit};
        return !journal_.unaccounted_references_since(tx_.call.token,spans,callbacks);
    }
    bool group_barrier() const {
        return stopped_ && std::all_of(targets_.begin(),targets_.end(),[](const auto& v){return v.second.stopped && v.second.drained;});
    }
public:
    Engine(Ops& ops,Journal& journal):ops_(ops),journal_(journal){}
    int reject(int code) {return fail(code);}
    std::vector<Key> target_keys() const {
        std::vector<Key> result;for(const auto& target:targets_)result.push_back(target.first);return result;
    }
    int open(const Config& c) {
        const auto owner_key=std::make_pair(c.owner_hi,c.owner_lo);
        if(installed_ || tx_.active || c.version!=1 || c.size!=sizeof(Config) || c.reserved
           || !c.owner_hi || !c.owner_lo || !c.generation || !c.thread || c.device<0
           || c.max_targets<1 || c.max_targets>512 || c.max_calls<1 || c.max_calls>8192
           || !c.max_device_bytes || c.max_device_bytes>64*1024*1024
           || !c.max_host_bytes || c.max_host_bytes>512*1024*1024
           || c.generation<=generations_[owner_key])return Identity;
        config_=c;installed_=true;stopped_=sealed_=false;error_=0;admitted_=child_prepares_=child_executes_=last_parent_=0;
        targets_.clear();generations_[owner_key]=c.generation;opened_sequence_=journal_.last_sequence();return Ok;
    }
    int admit(uint64_t parent_ticket,const Lease& lease) {
        const auto* call=journal_.parent_record(parent_ticket);const auto js=journal_.state();
        if(!installed_ || stopped_ || tx_.active)return Busy;
        if(js.errors || js.overflow || !call)return Observer;
        if(lease.version!=1 || lease.size!=sizeof(Lease) || lease.reserved || lease.mode<Reference || lease.mode>Off
           || lease.owner_hi!=config_.owner_hi || lease.owner_lo!=config_.owner_lo || lease.generation!=config_.generation
           || lease.thread!=config_.thread || call->thread!=config_.thread || call->device!=config_.device
           || !same_call(*call,lease.expected) || !valid_call(*call) || !key_valid(lease.target) || !key_valid(lease.actual_row_proof)
           || parent_ticket<=last_parent_)return Identity;
        if(admitted_>=config_.max_calls)return Budget;
        auto found=targets_.find(lease.target);
        if(found==targets_.end()) {
            if(lease.mode!=Reference || sealed_ || targets_.size()>=config_.max_targets)return Qualification;
            found=targets_.emplace(lease.target,Target{}).first;found->second.site=*call;
        } else if(!same_site(found->second.site,*call))return Geometry;
        auto& target=found->second;
        if(lease.mode==Reference && sealed_)return Qualification;
        if(lease.mode!=Reference && !target.reference)return Qualification;
        if(lease.mode==Fault && (!sealed_ || !target.lossless || !target.off || target.faults>=8))return Qualification;
        const uint64_t device=(lease.mode==Reference || lease.mode==Off)?0:
            (call->quant.bytes+call->scale.bytes)*(lease.mode==Fault?2:1);
        const uint64_t host=target_bytes()+5*call->input.bytes+8*call->quant.bytes+8*call->scale.bytes;
        if(device>config_.max_device_bytes || host>config_.max_host_bytes)return Budget;
        tx_=Transaction{};tx_.active=true;tx_.call=*call;tx_.lease=lease;
        auto& r=tx_.receipt;r.mode=lease.mode;r.owner_hi=lease.owner_hi;r.owner_lo=lease.owner_lo;r.generation=lease.generation;
        r.parent_token=parent_ticket;r.lifetime_token=call->lifetime_token;r.target=lease.target;r.actual_row_proof=lease.actual_row_proof;
        ++admitted_;last_parent_=parent_ticket;
        auto make=[&](ChildReceipt& c,uint32_t role){c.plan.role=role;c.plan.parent_token=parent_ticket;
            c.plan.child_lease=parent_ticket*4+role;c.plan.input=call->input;c.plan.quant=call->quant;c.plan.scale=call->scale;};
        make(r.effect,ModelEffect);
        if(lease.mode==Lossless || lease.mode==Fault) {
            if(!allocate(0,call->quant.bytes) || !allocate(1,call->scale.bytes))return fail(Allocation);
            r.effect.plan.quant.address=reinterpret_cast<uint64_t>(tx_.buffers[0]);r.effect.plan.quant.descriptor=0;
            r.effect.plan.scale.address=reinterpret_cast<uint64_t>(tx_.buffers[1]);r.effect.plan.scale.descriptor=0;
        }
        if(lease.mode==Fault) {
            make(r.reference,CorrectReference);
            if(!allocate(2,call->quant.bytes) || !allocate(3,call->scale.bytes))return fail(Allocation);
            r.reference.plan.quant.address=reinterpret_cast<uint64_t>(tx_.buffers[2]);r.reference.plan.quant.descriptor=0;
            r.reference.plan.scale.address=reinterpret_cast<uint64_t>(tx_.buffers[3]);r.reference.plan.scale.descriptor=0;
            r.effect.plan.input.rows=r.effect.plan.quant.rows=1;
            r.effect.plan.input.columns=r.effect.plan.quant.columns=call->input.rows*call->input.columns;
            r.effect.plan.input.stride0=r.effect.plan.quant.stride0=r.effect.plan.input.columns;
            r.effect.plan.input.descriptor=0;r.effect.plan.scale.rows=1;r.effect.plan.scale.bytes=4;
        }
        return Ok;
    }
    ChildPlan plan(uint32_t role) {auto* c=child(role);return c?c->plan:ChildPlan{};}
    int descriptor(uint32_t role,uint32_t formal,uint64_t handle) {
        auto* c=child(role);if(!tx_.active || !c || c->prepare_enter || formal>2 || !handle)return Identity;
        View* v=formal==0?&c->plan.input:formal==1?&c->plan.quant:&c->plan.scale;
        if(v->descriptor || handle==tx_.call.input.descriptor || handle==tx_.call.quant.descriptor || handle==tx_.call.scale.descriptor
           || std::find(tx_.descriptors.begin(),tx_.descriptors.end(),handle)!=tx_.descriptors.end())return fail(Identity);
        v->descriptor=handle;tx_.descriptors.push_back(handle);return Ok;
    }
    int bind_prepare(uint32_t role,uint64_t raw_ticket) {
        auto* c=child(role);const auto* e=journal_.event(raw_ticket);
        if(!tx_.active || !c || c->prepare_enter || !e || e->kind!=PrepareEnter || !e->registration
           || e->thread!=config_.thread || e->device!=config_.device
           || raw_ticket<=tx_.call.token || e->registration!=raw_ticket
           || e->smooth_tensor || e->group_index_tensor || e->offset_tensor || e->destination_dtype!=2
           || !same_view(e->input,c->plan.input) || !same_view(e->quant,c->plan.quant) || !same_view(e->scale,c->plan.scale)
           || !c->plan.input.descriptor || !c->plan.quant.descriptor || !c->plan.scale.descriptor)return fail(Identity);
        ChildBinding binding{};binding.parent_token=tx_.call.token;binding.child_lease=c->plan.child_lease;
        binding.owner_hi=config_.owner_hi;binding.owner_lo=config_.owner_lo;binding.generation=config_.generation;binding.role=role;
        if(!journal_.classify(raw_ticket,binding))return fail(Observer);
        c->prepare_enter=raw_ticket;c->registration=e->registration;++child_prepares_;return Ok;
    }
    int prepared(uint32_t role,uint64_t raw_ticket) {
        auto* c=child(role);const auto* e=journal_.event(raw_ticket);
        if(!tx_.active || !c || !c->prepare_enter || c->prepare_exit || !e || e->kind!=PrepareExit
           || raw_ticket<=c->prepare_enter || e->registration!=c->registration || e->thread!=config_.thread || e->device!=config_.device)return fail(Identity);
        c->prepare_exit=raw_ticket;c->executor=e->executor;c->prepare_status=e->status;c->prepare_original_calls=e->original_calls;
        c->workspace_required=e->workspace_required;
        c->prepared=e->executor!=0; // Even failed GWS with a nonnull executor must remain owned.
        if(e->status || e->original_calls!=1 || !e->executor)return fail(Native);
        if(role==CorrectReference && c->workspace_required) {
            if(c->workspace_required>config_.max_device_bytes-owned_device_bytes())return fail(Budget);
            if(!allocate_owned(tx_.helper_workspace,tx_.helper_workspace_bytes,c->workspace_required))return fail(Allocation);
            c->workspace=reinterpret_cast<uint64_t>(tx_.helper_workspace);
            c->workspace_capacity=c->workspace_required;c->workspace_group_owned=1;
        }
        return Ok;
    }
    int start(uint64_t stream,uint64_t workspace=0,uint64_t capacity=0) {
        if(!tx_.active || tx_.started || !stream || !tx_.receipt.effect.prepared || tx_.receipt.error)return StateError;
        if(tx_.lease.mode==Fault && !tx_.receipt.reference.prepared)return StateError;
        if(capacity<tx_.receipt.effect.workspace_required || (capacity && !span(workspace,capacity)) || (!capacity && workspace))return fail(Geometry);
        if(capacity>config_.max_device_bytes-owned_device_bytes())return fail(Budget);
        if(capacity && !disjoint_from_resources(workspace,capacity))return fail(Geometry);
        tx_.borrowed_workspace=workspace;tx_.borrowed_workspace_bytes=capacity;
        tx_.receipt.effect.workspace=workspace;tx_.receipt.effect.workspace_capacity=capacity;
        tx_.stream=stream;
        if(!ops_.sync(stream))return fail(Sync);
        if(!read(tx_.input_before,tx_.call.input.address,tx_.call.input.bytes)
           || !read(tx_.public_quant_before,tx_.call.quant.address,tx_.call.quant.bytes)
           || !read(tx_.public_scale_before,tx_.call.scale.address,tx_.call.scale.bytes))return fail(Copy);
        tx_.started=true;tx_.receipt.input_before_captured=1;
        const auto& target=targets_.at(tx_.lease.target);
        if(tx_.lease.mode!=Fault && target.reference && tx_.input_before!=target.input)return fail(Qualification);
        return Ok;
    }
    int bind_execute(uint32_t role,uint64_t raw_ticket) {
        auto* c=child(role);const auto* e=journal_.event(raw_ticket);
        if(!tx_.active || !tx_.started || !c || !c->prepared || c->execute_enter || !e || e->kind!=ExecuteEnter
           || e->registration!=c->registration || e->executor!=c->executor || e->thread!=config_.thread
           || raw_ticket<=c->prepare_exit || e->device!=config_.device || e->stream!=tx_.stream || !e->stream
           || e->workspace!=c->workspace || e->workspace_capacity!=c->workspace_capacity)return fail(Identity);
        if(role==ModelEffect && tx_.lease.mode==Fault && !tx_.receipt.reference.readback)return fail(StateError);
        c->execute_enter=raw_ticket;c->stream=e->stream;++child_executes_;return Ok;
    }
    int executed(uint32_t role,uint64_t raw_ticket) {
        auto* c=child(role);const auto* e=journal_.event(raw_ticket);
        if(!tx_.active || !c || !c->execute_enter || c->execute_exit || !e || e->kind!=ExecuteExit
           || e->registration!=c->registration || e->executor!=c->executor || e->thread!=config_.thread
           || raw_ticket<=c->execute_enter || e->device!=config_.device || e->stream!=c->stream
           || e->workspace!=c->workspace || e->workspace_capacity!=c->workspace_capacity)return fail(Identity);
        c->execute_exit=raw_ticket;c->execute_original_calls=e->original_calls;c->execute_status=e->status;c->returned=1;
        if(e->status || e->original_calls!=1)return fail(Native);
        return Ok;
    }
    int readback(uint32_t role) {
        auto* c=child(role);
        if(!tx_.active || !c || !c->returned || c->execute_status || c->execute_original_calls!=1 || c->readback)return StateError;
        tx_.receipt.input_unchanged=0;
        if(!ops_.sync(c->stream))return fail(Sync);
        c->drained=1;
        if(!unchanged())return fail(InputChanged);
        tx_.receipt.input_unchanged=1;
        auto& q=role==CorrectReference?tx_.reference_quant:tx_.effect_quant;
        auto& s=role==CorrectReference?tx_.reference_scale:tx_.effect_scale;
        if(role==ModelEffect && tx_.lease.mode==Fault) {
            for(int64_t row=1;row<tx_.call.input.rows;++row) {
                if(!ops_.copy(reinterpret_cast<void*>(c->plan.scale.address+uint64_t(row)*4),4,
                    reinterpret_cast<void*>(c->plan.scale.address),4,CopyKind::DeviceToDevice))return fail(Copy);
            }
            if(!ops_.sync(c->stream))return fail(Sync);
        }
        if(!read(q,c->plan.quant.address,tx_.call.quant.bytes) || !read(s,c->plan.scale.address,tx_.call.scale.bytes))return fail(Copy);
        c->readback=1;return Ok;
    }
    int publish() {
        auto& r=tx_.receipt;
        if(!tx_.active || !r.effect.readback || r.error || tx_.published)return StateError;
        if(tx_.lease.mode==Fault) {
            if(!r.reference.readback)return fail(Qualification);
            r.reference_effect_compared=1;
            r.changed_quant_bytes=differences(tx_.reference_quant,tx_.effect_quant);
            r.changed_scale_bytes=differences(tx_.reference_scale,tx_.effect_scale);
            if(!r.changed_quant_bytes && !r.changed_scale_bytes)return fail(NoEffect);
        } else if(targets_.at(tx_.lease.target).reference) {
            const auto& target=targets_.at(tx_.lease.target);
            if(tx_.effect_quant!=target.quant || tx_.effect_scale!=target.scale)return fail(Qualification);
        }
        if(!unchanged())return fail(InputChanged);
        // Set before the first possible write: partial publication needs retryable restoration.
        tx_.published=true;
        if(tx_.lease.mode==Lossless || tx_.lease.mode==Fault) {
            if(!write_exact(tx_.call.quant.address,tx_.effect_quant) || !write_exact(tx_.call.scale.address,tx_.effect_scale))return fail(Copy);
        }
        if(!ops_.sync(tx_.stream)){r.input_unchanged=0;return fail(Sync);}
        if(!unchanged())return fail(InputChanged);
        r.published=r.publication_exact=1;return Ok;
    }
    int acknowledge() {
        if(!tx_.active || !tx_.receipt.publication_exact || (tx_.receipt.error && tx_.receipt.error!=Free))return StateError;
        if(!tx_.receipt.effect.drained || (tx_.lease.mode==Fault && !tx_.receipt.reference.drained))return Pending;
        if(!raw_quiescent() || (stopped_ && !group_barrier()))return Pending;
        if(release_owned())return Free;
        auto& target=targets_.at(tx_.lease.target);target.last_stream=tx_.stream;
        if(tx_.lease.mode==Reference && !target.reference) {
            target.input=std::move(tx_.input_before);target.quant=std::move(tx_.effect_quant);target.scale=std::move(tx_.effect_scale);
            target.reference=true;
        } else if(tx_.lease.mode==Lossless)target.lossless=true;
        else if(tx_.lease.mode==Off)target.off=true;
        else if(tx_.lease.mode==Fault)++target.faults;
        tx_.receipt.released=1;last_=tx_.receipt;tx_=Transaction{};return Ok;
    }
    int seal(const std::vector<Key>& keys,uint64_t unresolved_candidates,const Key& complete_scope) {
        if(!installed_ || stopped_ || tx_.active || sealed_ || unresolved_candidates || !key_valid(complete_scope)
           || keys.empty() || keys.size()!=targets_.size() || std::set<Key>(keys.begin(),keys.end()).size()!=keys.size())return Qualification;
        for(const auto& key:keys){auto t=targets_.find(key);if(t==targets_.end() || !t->second.reference || !t->second.lossless || !t->second.off)return Qualification;}
        sealed_=true;return Ok;
    }
    int stop(const Config& c,const Key& target) {
        if(!owner(c) || !targets_.count(target))return Identity;
        stopped_=true;targets_.at(target).stopped=true;return Ok;
    }
    int cancel_empty(const Config& c) {
        if(!owner(c) || !targets_.empty() || tx_.active || admitted_ || sealed_)return Identity;
        stopped_=true; // Retry cannot admit a target while a raw call is unresolved.
        if(!raw_quiescent())return Pending;
        std::set<uint64_t> streams;
        for(uint64_t sequence=opened_sequence_+1;sequence<=journal_.last_sequence();++sequence) {
            const auto* event=journal_.event(sequence);
            if(!event || event->kind!=ExecuteExit)continue;
            if(event->status || event->original_calls!=1 || !event->stream
               || event->thread!=config_.thread || event->device!=config_.device)return Pending;
            streams.insert(event->stream);
        }
        // Raw pairing is not device drain. Every actual passthrough stream
        // seen under this owner must complete before its callbacks can go away.
        for(const auto stream:streams)if(!ops_.sync(stream))return Pending;
        installed_=false;return Ok;
    }
    bool drain(const Config& c,const Key& key) {
        if(!owner(c) || !targets_.count(key) || !raw_quiescent()
           || !std::all_of(targets_.begin(),targets_.end(),[](const auto& v){return v.second.stopped;}))return false;
        auto& target=targets_.at(key);target.drained=false;
        if(tx_.active && tx_.lease.target==key) {
            for(auto* child_record:{&tx_.receipt.reference,&tx_.receipt.effect}) {
                if(!child_record->prepared)continue;
                if(!child_record->returned || !child_record->stream || !ops_.sync(child_record->stream))return false;
                child_record->drained=1;
            }
            if(tx_.started && !ops_.sync(tx_.stream))return false;
        }
        if(target.last_stream && !ops_.sync(target.last_stream))return false;
        target.drained=true;return true;
    }
    int recover(const Config& c,const LiveWitness* live=nullptr) {
        if(!owner(c) || !group_barrier() || !raw_quiescent())return Pending;
        if(tx_.active) {
            if(tx_.started) {
                if(!live || live->parent_token!=tx_.call.token || !live->lifetime_token || live->lifetime_token!=tx_.call.lifetime_token
                   || !same_view(live->input,tx_.call.input) || !same_view(live->quant,tx_.call.quant)
                   || !same_view(live->scale,tx_.call.scale))return fail(Liveness);
                if(!unchanged())return fail(InputChanged);
            }
            const bool public_written=tx_.published || (tx_.receipt.effect.execute_enter &&
                (tx_.lease.mode==Reference || tx_.lease.mode==Off));
            if(public_written && !tx_.restored) {
                if(!tx_.started || !write_exact(tx_.call.quant.address,tx_.public_quant_before)
                   || !write_exact(tx_.call.scale.address,tx_.public_scale_before) || !ops_.sync(tx_.stream))return fail(Restore);
                tx_.restored=true;tx_.receipt.public_bytes_restored=1; // Byte rollback, NOT healthy OFF.
            }
            if(tx_.started && !unchanged())return fail(InputChanged);
            if(release_owned())return Free;
            tx_.receipt.released=1;last_=tx_.receipt;tx_=Transaction{};
        }
        // All stop/drain gates preceded every target-reference release.
        targets_.clear();installed_=false;return Ok;
    }
    Receipt receipt() const {return tx_.active?tx_.receipt:last_;}
    State state() const {
        State s{};s.installed=installed_;s.stopped=stopped_;s.sealed=sealed_;s.owner_hi=config_.owner_hi;s.owner_lo=config_.owner_lo;
        s.generation=config_.generation;s.targets=targets_.size();s.admitted_model_calls=admitted_;s.pending_parent=tx_.active;
        s.native_child_prepares=child_prepares_;s.native_child_executes=child_executes_;s.error=error_;
        s.host_bytes=target_bytes()+tx_.host_bytes();s.device_bytes=owned_device_bytes();
        s.owned_workspace_bytes=tx_.helper_workspace_bytes;s.borrowed_workspace_bytes=tx_.borrowed_workspace_bytes;
        s.quarantined_bytes=tx_.receipt.unsafe_allocation_bytes;
        for(auto d:tx_.descriptors)s.owned_descriptors+=d!=0;
        for(const auto& v:targets_){s.stopped_targets+=v.second.stopped;s.drained_targets+=v.second.drained;}
        return s;
    }
};
} // namespace dcat_dq_group
#endif
