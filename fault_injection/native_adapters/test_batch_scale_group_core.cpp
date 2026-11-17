// Explicit CPU memory simulation. No CANN/NPU execution or rounding oracle.
#include "batch_scale_group_core.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
using namespace dcat_dq_group;
namespace {
struct CpuOps final:Ops {
    std::map<void*,size_t> memory;std::vector<std::string> actions;
    uint64_t sync_failure{},copy_fail_at{},copy_calls{},release_failures{},descriptor_failures{};
    void* alias_next_allocation{};
    void* fail_release_pointer{};
    ~CpuOps(){for(const auto& p:memory)std::free(p.first);}
    bool allocate(void** p,size_t bytes) override {
        if(alias_next_allocation){*p=alias_next_allocation;alias_next_allocation=nullptr;return true;}
        *p=std::malloc(bytes);if(*p)memory[*p]=bytes;return *p!=nullptr;
    }
    bool release(void* p) override {
        actions.push_back("release");if(release_failures){--release_failures;return false;}
        if(p==fail_release_pointer){fail_release_pointer=nullptr;return false;}
        auto it=memory.find(p);assert(it!=memory.end());std::free(p);memory.erase(it);return true;
    }
    bool copy(void* output,size_t capacity,const void* input,size_t bytes,CopyKind) override {
        ++copy_calls;if(copy_fail_at && copy_calls==copy_fail_at)return false;
        assert(capacity>=bytes);std::memmove(output,input,bytes);return true;
    }
    bool sync(uint64_t stream) override {actions.push_back("sync:"+std::to_string(stream));return stream!=sync_failure;}
    bool destroy_descriptor(uint64_t d) override {
        assert(d);actions.push_back("descriptor");if(descriptor_failures){--descriptor_failures;return false;}return true;
    }
};
struct Fixture {
    CpuOps ops;Journal journal;Engine engine{ops,journal};Config config{};
    std::array<uint16_t,8> input{{1,2,3,4,5,6,7,8}};
    std::array<int8_t,8> quant{{-9,-9,-9,-9,-9,-9,-9,-9}};
    std::array<float,2> scale{{-9,-9}};
    uint64_t descriptor_id{10000},executor_id{20000},lifetime_id{30000};Call call{};
    Key a{{1,2,3,4}},b{{5,6,7,8}},row_proof{{11,12,13,14}};
    explicit Fixture(uint64_t device_budget=1<<20){config.max_targets=8;config.max_calls=64;config.owner_hi=7;config.owner_lo=9;config.generation=1;
        config.thread=41;config.device=0;config.max_device_bytes=device_budget;config.max_host_bytes=1<<20;
        assert(engine.open(config)==Ok);}
    static View view(uint64_t address,uint64_t bytes,uint64_t descriptor,int64_t rows,int64_t columns,int dtype) {
        View v{};v.address=address;v.bytes=bytes;v.descriptor=descriptor;v.rows=rows;v.columns=columns;
        v.dimensions=columns?2:1;v.dtype=dtype;v.format=2;v.stride0=columns?columns:1;v.stride1=columns?1:0;return v;
    }
    Call observe() {
        Call c{};c.thread=config.thread;c.device=config.device;c.lifetime_token=++lifetime_id;
        c.input=view(reinterpret_cast<uint64_t>(input.data()),sizeof(input),10,2,4,27);
        c.quant=view(reinterpret_cast<uint64_t>(quant.data()),sizeof(quant),11,2,4,2);
        c.scale=view(reinterpret_cast<uint64_t>(scale.data()),sizeof(scale),12,2,0,0);
        call=journal.parent(c);return call;
    }
    Lease lease(Mode mode,const Key& target) {
        Lease l{};l.mode=mode;l.owner_hi=config.owner_hi;l.owner_lo=config.owner_lo;l.generation=config.generation;
        l.thread=config.thread;l.target=target;l.actual_row_proof=row_proof;l.expected=call;return l;
    }
    int admit(Mode mode,const Key& target) {observe();return engine.admit(call.token,lease(mode,target));}
    ChildPlan descriptors(Role role) {
        auto p=engine.plan(role);
        for(unsigned i=0;i<3;++i){const View& v=i==0?p.input:i==1?p.quant:p.scale;
            if(!v.descriptor)assert(engine.descriptor(role,i,++descriptor_id)==Ok);}
        return engine.plan(role);
    }
    uint64_t prepare(Role role,uint64_t executor=0,int status=0,uint64_t workspace_required=0) {
        const auto p=descriptors(role);
        const auto entry=journal.prepare(p.input,p.quant,p.scale,config.thread,config.device);
        assert(engine.bind_prepare(role,entry)==Ok);
        const auto exit=journal.prepared(entry,executor?executor:++executor_id,status,1,workspace_required);
        const int code=engine.prepared(role,exit);assert(code==(status?Native:Ok));return executor?executor:executor_id;
    }
    // These bytes stand for independently observed real Native results. The
    // core is not allowed to compute or guess DynamicQuant rounding.
    void native_outputs(Role role,bool fault_values) {
        const auto p=engine.plan(role);const int8_t good[8]{1,2,3,4,5,6,7,8};const int8_t bad[8]{1,1,2,2,3,3,4,4};
        const float good_scale[2]{.25f,.5f},bad_scale[1]{.5f};
        std::memcpy(reinterpret_cast<void*>(p.quant.address),fault_values?bad:good,8);
        std::memcpy(reinterpret_cast<void*>(p.scale.address),fault_values?bad_scale:good_scale,fault_values?4:8);
    }
    void execute(Role role,uint64_t executor,uint64_t stream,bool fault_values,int status=0) {
        const auto receipt=engine.receipt();const auto& child=role==CorrectReference?receipt.reference:receipt.effect;
        const auto enter=journal.execute(executor,config.thread,config.device,stream,child.workspace,child.workspace_capacity);
        assert(engine.bind_execute(role,enter)==Ok);native_outputs(role,fault_values);
        const auto exit=journal.executed(enter,status,1);
        assert(engine.executed(role,exit)==(status?Native:Ok));
    }
    Receipt finish(Mode mode,const Key& target,uint64_t stream=71) {
        assert(admit(mode,target)==Ok);
        const auto ref=mode==Fault?prepare(CorrectReference):0;
        const auto effect=prepare(ModelEffect);
        assert(engine.start(stream)==Ok);
        if(ref){execute(CorrectReference,ref,stream,false);assert(engine.readback(CorrectReference)==Ok);}
        execute(ModelEffect,effect,stream,mode==Fault);assert(engine.readback(ModelEffect)==Ok);
        assert(engine.publish()==Ok);assert(engine.acknowledge()==Ok);return engine.receipt();
    }
    void qualify(const Key& key,uint64_t stream=71) {finish(Reference,key,stream);finish(Lossless,key,stream);finish(Off,key,stream);}
    void seal(std::vector<Key> keys) {assert(engine.seal(keys,0,row_proof)==Ok);}
    void stop_drain(const std::vector<Key>& keys) {
        for(const auto& k:keys)assert(engine.stop(config,k)==Ok);
        for(const auto& k:keys)assert(engine.drain(config,k));
    }
    LiveWitness witness() const {LiveWitness w{};w.parent_token=call.token;w.lifetime_token=call.lifetime_token;
        w.input=call.input;w.quant=call.quant;w.scale=call.scale;return w;}
    int recover_live() {const auto live=witness();return engine.recover(config,&live);}
};
void complete_two_targets_actual_reference_effect() {
    Fixture f;f.qualify(f.a,71);f.qualify(f.b,72);f.seal({f.a,f.b});
    const auto before=f.journal.state();f.input[7]=999; // Actual F input differs from healthy B.
    const auto r=f.finish(Fault,f.a,73);const auto after=f.journal.state();
    assert(after.total-before.total==9 && after.model_parents-before.model_parents==1 && after.native_prepares-before.native_prepares==2
        && after.native_executes-before.native_executes==2 && after.helper_prepares-before.helper_prepares==1);
    assert(r.reference.prepare_original_calls==1 && r.effect.prepare_original_calls==1
        && r.reference.execute_original_calls==1 && r.effect.execute_original_calls==1);
    assert(r.reference.registration!=r.effect.registration && r.reference_effect_compared && r.changed_quant_bytes
        && r.changed_scale_bytes && r.input_unchanged && r.published && r.released && f.input[7]==999);
    assert(f.scale[0]==.5f && f.scale[1]==.5f);
    f.input[7]=8;f.finish(Off,f.a,73);f.finish(Off,f.b,72);
    f.stop_drain({f.a,f.b});assert(f.engine.recover(f.config)==Ok);
    const auto s=f.engine.state();assert(!s.installed && !s.pending_parent && !s.device_bytes && !s.host_bytes);
}
void missing_helper_wrong_identity_and_no_early_publish() {
    Fixture f;f.qualify(f.a);f.seal({f.a});assert(f.admit(Fault,f.a)==Ok);
    f.prepare(ModelEffect);assert(f.engine.start(71)==StateError);assert(f.engine.publish()==StateError);
    assert(f.engine.stop(f.config,f.a)==Ok);assert(!f.engine.drain(f.config,f.a));assert(f.engine.recover(f.config)==Pending);
    assert(f.engine.state().pending_parent && f.engine.state().device_bytes && f.ops.actions.back()!="release");
}
void wrong_child_role_keeps_raw_census() {
    Fixture f;f.qualify(f.a);f.seal({f.a});assert(f.admit(Fault,f.a)==Ok);
    const auto p=f.descriptors(CorrectReference);f.descriptors(ModelEffect);
    const auto raw=f.journal.prepare(p.input,p.quant,p.scale,f.config.thread,0);
    const auto before=f.journal.state().native_prepares;
    assert(f.engine.bind_prepare(ModelEffect,raw)==Identity);
    assert(f.journal.state().native_prepares==before && f.journal.state().unclassified_prepares==1);
}
void prepared_failure_with_executor_never_drains() {
    Fixture f;f.qualify(f.a);f.seal({f.a});assert(f.admit(Fault,f.a)==Ok);
    f.prepare(CorrectReference);f.prepare(ModelEffect,555,17);
    assert(f.engine.stop(f.config,f.a)==Ok);assert(!f.engine.drain(f.config,f.a));
    assert(f.engine.recover(f.config)==Pending && f.engine.state().owned_descriptors);
}
void no_effect_and_last_input_byte_are_not_hits() {
    for(bool mutate:{false,true}){
        Fixture f;f.qualify(f.a);f.seal({f.a});assert(f.admit(Fault,f.a)==Ok);
        auto ref=f.prepare(CorrectReference),effect=f.prepare(ModelEffect);assert(f.engine.start(71)==Ok);
        f.execute(CorrectReference,ref,71,!mutate);
        if(!mutate){const float actual_reference_scale[2]{.5f,.5f};
            std::memcpy(reinterpret_cast<void*>(f.engine.plan(CorrectReference).scale.address),actual_reference_scale,8);}
        assert(f.engine.readback(CorrectReference)==Ok);
        f.execute(ModelEffect,effect,71,true);
        if(mutate)reinterpret_cast<unsigned char*>(f.input.data())[sizeof(f.input)-1]^=1;
        assert(f.engine.readback(ModelEffect)==(mutate?InputChanged:Ok));
        if(!mutate){
            assert(f.engine.publish()==NoEffect);
            const auto r=f.engine.receipt();assert(r.reference_effect_compared && !r.changed_quant_bytes && !r.changed_scale_bytes);
            assert(!r.published && !r.released && f.engine.state().device_bytes);
            f.stop_drain({f.a});assert(f.engine.recover(f.config)==Liveness);
            assert(f.engine.receipt().error==NoEffect && f.engine.receipt().last_error==Liveness);
            assert(f.recover_live()==Ok && f.engine.receipt().error==NoEffect);
        } else {
            assert(f.engine.publish()==StateError && !f.engine.receipt().published && !f.engine.receipt().input_unchanged);
            f.stop_drain({f.a});f.ops.actions.clear();assert(f.recover_live()==InputChanged);
            assert(f.ops.actions.empty() && f.engine.state().device_bytes);
        }
    }
}
void stream_failure_and_publish_failure_retain_retry() {
    Fixture f;f.qualify(f.a);f.seal({f.a});assert(f.admit(Fault,f.a)==Ok);
    const auto ref=f.prepare(CorrectReference),effect=f.prepare(ModelEffect);assert(f.engine.start(71)==Ok);
    f.execute(CorrectReference,ref,71,false);assert(f.engine.readback(CorrectReference)==Ok);
    f.execute(ModelEffect,effect,71,true);f.ops.sync_failure=71;
    assert(f.engine.readback(ModelEffect)==Sync);assert(f.engine.stop(f.config,f.a)==Ok);
    assert(!f.engine.drain(f.config,f.a));assert(f.engine.recover(f.config)==Pending);
    f.ops.sync_failure=0;assert(f.engine.drain(f.config,f.a));assert(f.recover_live()==Ok);
    assert(!f.engine.state().device_bytes);

    Fixture g;g.qualify(g.a);g.seal({g.a});assert(g.admit(Fault,g.a)==Ok);
    const auto before_q=g.quant;const auto before_s=g.scale;
    const auto rr=g.prepare(CorrectReference),ee=g.prepare(ModelEffect);assert(g.engine.start(71)==Ok);
    g.execute(CorrectReference,rr,71,false);assert(g.engine.readback(CorrectReference)==Ok);
    g.execute(ModelEffect,ee,71,true);assert(g.engine.readback(ModelEffect)==Ok);
    g.ops.copy_fail_at=g.ops.copy_calls+4; // Input read, Q write/read, then fail scale write.
    assert(g.engine.publish()==Copy);assert(!g.engine.receipt().publication_exact);
    g.stop_drain({g.a});assert(g.engine.recover(g.config)==Liveness);
    auto live=g.witness();g.ops.copy_fail_at=0;g.ops.release_failures=1;
    assert(g.engine.recover(g.config,&live)==Free);assert(g.engine.state().pending_parent);
    assert(g.quant==before_q && g.scale==before_s);
    assert(g.engine.recover(g.config,&live)==Ok);assert(!g.engine.state().device_bytes);
}
void whole_group_barrier_and_complete_qualification() {
    Fixture f;f.qualify(f.a,71);f.finish(Reference,f.b,72);f.finish(Lossless,f.b,72);
    assert(f.engine.seal({f.a,f.b},0,f.row_proof)==Qualification);
    f.finish(Off,f.b,72);assert(f.engine.seal({f.a},0,f.row_proof)==Qualification);
    assert(f.engine.seal({f.a,f.b},1,f.row_proof)==Qualification);f.seal({f.a,f.b});
    const auto host=f.engine.state().host_bytes;
    assert(f.engine.stop(f.config,f.a)==Ok);assert(f.engine.stop(f.config,f.b)==Ok);
    assert(f.engine.drain(f.config,f.a));f.ops.sync_failure=72;assert(!f.engine.drain(f.config,f.b));
    f.ops.actions.clear();assert(f.engine.recover(f.config)==Pending);
    assert(f.ops.actions.empty() && f.engine.state().host_bytes==host);
    f.ops.sync_failure=0;assert(f.engine.drain(f.config,f.b));assert(f.engine.recover(f.config)==Ok);
}
void executor_reuse_owner_generation_and_budget() {
    Fixture f;f.observe();auto lease=f.lease(Reference,f.a);lease.generation++;
    assert(f.engine.admit(f.call.token,lease)==Identity);
    lease=f.lease(Reference,f.a);lease.owner_lo++;assert(f.engine.admit(f.call.token,lease)==Identity);
    assert(f.engine.admit(f.call.token,f.lease(Reference,f.a))==Ok);
    const auto executor=f.prepare(ModelEffect,900);assert(f.engine.start(71)==Ok);
    f.execute(ModelEffect,executor,71,false);assert(f.engine.readback(ModelEffect)==Ok);
    auto first=f.engine.receipt().effect.registration;
    assert(f.engine.publish()==Ok);assert(f.engine.acknowledge()==Ok);
    assert(f.admit(Lossless,f.a)==Ok);f.prepare(ModelEffect,900);assert(f.engine.start(71)==Ok);
    f.execute(ModelEffect,900,71,false);assert(f.engine.readback(ModelEffect)==Ok);
    assert(f.engine.receipt().effect.registration!=first);assert(f.engine.publish()==Ok);assert(f.engine.acknowledge()==Ok);
    f.stop_drain({f.a});assert(f.engine.recover(f.config)==Ok);assert(f.engine.open(f.config)==Identity);
    f.config.generation++;f.config.max_calls=1;assert(f.engine.open(f.config)==Ok);f.finish(Reference,f.a);
    assert(f.admit(Lossless,f.a)==Budget);
}
bool has_release(const CpuOps& ops) {
    return std::any_of(ops.actions.begin(),ops.actions.end(),[](const auto& a){return a=="release" || a=="descriptor";});
}
void unbound_and_missing_callbacks_never_release() {
    for(unsigned stage:{0u,1u,2u,3u}) {
        Fixture f;f.qualify(f.a);f.seal({f.a});assert(f.admit(Fault,f.a)==Ok);
        const auto p=f.descriptors(CorrectReference);
        const auto enter=f.journal.prepare(p.input,p.quant,p.scale,f.config.thread,0);
        if(stage==0)assert(f.engine.bind_prepare(CorrectReference,enter)==Ok); // GWS still in progress.
        else {
            const auto exit=f.journal.prepared(enter,909,0,1);
            if(stage==1)assert(f.engine.bind_prepare(ModelEffect,enter)==Identity);
            if(stage>=2) {
                // The observer sees actual calls even when adapter callbacks
                // are absent or reject an incompatible parent/role.
                if(stage==3){assert(f.engine.bind_prepare(CorrectReference,enter)==Ok);
                    assert(f.engine.prepared(ModelEffect,exit)==Identity);}
                const auto execute=f.journal.execute(909,f.config.thread,0,71);
                f.journal.executed(execute,0,1);
                assert(!f.journal.state().pending && !f.journal.state().executing);
            }
        }
        assert(f.engine.stop(f.config,f.a)==Ok);f.ops.actions.clear();
        assert(!f.engine.drain(f.config,f.a));assert(f.engine.recover(f.config)==Pending);
        assert(!has_release(f.ops) && f.engine.state().device_bytes && f.engine.state().owned_descriptors);
    }
}
void old_raw_ticket_cannot_attach_to_new_parent() {
    Fixture f;const auto old=f.observe();
    const auto enter=f.journal.prepare(old.input,old.quant,old.scale,f.config.thread,0);
    f.journal.prepared(enter,700,0,1);
    assert(f.admit(Reference,f.a)==Ok);assert(enter<f.call.token);
    assert(f.engine.bind_prepare(ModelEffect,enter)==Identity);
    assert(f.journal.state().unclassified_prepares==1 && f.journal.state().pending==1);
    assert(f.engine.stop(f.config,f.a)==Ok);assert(!f.engine.drain(f.config,f.a));
}
void repeated_reference_cannot_replace_frozen_qualification() {
    Fixture f;f.qualify(f.a);f.finish(Reference,f.a); // Exact repeat is read-only revalidation.
    f.seal({f.a});const auto admitted=f.engine.state().admitted_model_calls;
    assert(f.admit(Reference,f.a)==Qualification);
    assert(f.engine.state().admitted_model_calls==admitted && !f.engine.state().pending_parent);
    f.finish(Fault,f.a);f.stop_drain({f.a});assert(f.engine.recover(f.config)==Ok);

    Fixture g;g.qualify(g.a);g.input[7]=999;
    assert(g.admit(Reference,g.a)==Ok);g.prepare(ModelEffect);
    assert(g.engine.start(71)==Qualification);assert(g.engine.seal({g.a},0,g.row_proof)==Qualification);
    assert(g.admit(Fault,g.a)==Busy);assert(g.engine.state().pending_parent);

    Fixture h;h.qualify(h.a);assert(h.admit(Reference,h.a)==Ok);
    auto exec=h.prepare(ModelEffect);assert(h.engine.start(71)==Ok);
    h.execute(ModelEffect,exec,71,true);assert(h.engine.readback(ModelEffect)==Ok);
    assert(h.engine.publish()==Qualification);assert(!h.engine.receipt().published);
    assert(h.engine.seal({h.a},0,h.row_proof)==Qualification);
    assert(h.admit(Fault,h.a)==Busy);
}
void stopped_ack_waits_for_all_target_drains() {
    Fixture f;f.qualify(f.a,71);f.qualify(f.b,72);f.seal({f.a,f.b});
    assert(f.admit(Fault,f.a)==Ok);auto ref=f.prepare(CorrectReference),effect=f.prepare(ModelEffect);
    assert(f.engine.start(71)==Ok);f.execute(CorrectReference,ref,71,false);
    assert(f.engine.readback(CorrectReference)==Ok);f.execute(ModelEffect,effect,71,true);
    assert(f.engine.readback(ModelEffect)==Ok && f.engine.publish()==Ok);
    assert(f.engine.stop(f.config,f.a)==Ok);f.ops.actions.clear();
    assert(f.engine.acknowledge()==Pending && !has_release(f.ops));
    assert(f.engine.stop(f.config,f.b)==Ok);assert(f.engine.drain(f.config,f.a));
    f.ops.sync_failure=72;assert(!f.engine.drain(f.config,f.b));
    assert(f.engine.acknowledge()==Pending && f.engine.recover(f.config)==Pending && !has_release(f.ops));
    f.ops.sync_failure=0;assert(f.engine.drain(f.config,f.b));
    const auto live=f.witness();assert(f.engine.recover(f.config,&live)==Ok);
}
void workspace_has_exact_call_binding_and_ownership() {
    Fixture f;f.qualify(f.a);f.seal({f.a});assert(f.admit(Fault,f.a)==Ok);
    const auto ref=f.prepare(CorrectReference,0,0,128),effect=f.prepare(ModelEffect,0,0,64);
    std::array<unsigned char,64> borrowed{};
    assert(f.engine.start(71,reinterpret_cast<uint64_t>(borrowed.data()),borrowed.size())==Ok);
    auto r=f.engine.receipt();auto state=f.engine.state();
    assert(r.reference.workspace_required==128 && r.reference.workspace_group_owned && r.reference.workspace_capacity==128);
    assert(r.effect.workspace_required==64 && !r.effect.workspace_group_owned && r.effect.workspace==reinterpret_cast<uint64_t>(borrowed.data()));
    assert(state.device_bytes==160 && state.owned_workspace_bytes==128 && state.borrowed_workspace_bytes==64);
    f.execute(CorrectReference,ref,71,false);assert(f.engine.readback(CorrectReference)==Ok);
    f.execute(ModelEffect,effect,71,true);assert(f.engine.readback(ModelEffect)==Ok);
    r=f.engine.receipt();const auto* raw=f.journal.event(r.effect.execute_enter);
    assert(raw && raw->workspace_required==64 && raw->workspace==r.effect.workspace && raw->workspace_capacity==64);
    assert(f.engine.publish()==Ok);
    f.ops.fail_release_pointer=reinterpret_cast<void*>(r.reference.workspace);
    assert(f.engine.acknowledge()==Free && !f.engine.receipt().released);
    assert(f.engine.state().device_bytes==128 && f.engine.state().owned_workspace_bytes==128
        && f.engine.state().borrowed_workspace_bytes==64);
    assert(f.engine.acknowledge()==Ok);
    assert(f.ops.memory.empty() && !f.engine.state().device_bytes && !f.engine.state().borrowed_workspace_bytes);
    assert(std::all_of(borrowed.begin(),borrowed.end(),[](unsigned char n){return n==0;}));
    f.stop_drain({f.a});assert(f.engine.recover(f.config)==Ok);
}
void workspace_budget_alias_and_wrong_pointer_refuse() {
    for(unsigned failure:{0u,1u,2u,3u}) {
        Fixture f(failure==0?96:1024);f.qualify(f.a);f.seal({f.a});assert(f.admit(Fault,f.a)==Ok);
        auto ref=f.prepare(CorrectReference,0,0,64);f.prepare(ModelEffect,0,0,16);
        std::array<unsigned char,64> borrowed{},other{};
        const uint64_t pointer=failure==1?f.call.input.address:reinterpret_cast<uint64_t>(borrowed.data());
        const uint64_t capacity=failure==2?8:16;
        const int started=f.engine.start(71,pointer,capacity);
        assert(started==(failure==0?Budget:failure<3?Geometry:Ok));
        if(failure==3) {
            auto r=f.engine.receipt();const auto raw=f.journal.execute(ref,f.config.thread,0,71,reinterpret_cast<uint64_t>(other.data()),64);
            assert(r.reference.workspace!=reinterpret_cast<uint64_t>(other.data()));
            assert(f.engine.bind_execute(CorrectReference,raw)==Identity);
            f.journal.executed(raw,0,1);
        }
        assert(f.engine.stop(f.config,f.a)==Ok);f.ops.actions.clear();
        assert(!f.engine.drain(f.config,f.a) && f.engine.recover(f.config)==Pending && !has_release(f.ops));
        assert(f.engine.state().owned_workspace_bytes==64);
    }
    Fixture g(64);g.qualify(g.a);g.seal({g.a});assert(g.admit(Fault,g.a)==Ok);
    const auto p=g.descriptors(CorrectReference);const auto entry=g.journal.prepare(p.input,p.quant,p.scale,g.config.thread,0);
    assert(g.engine.bind_prepare(CorrectReference,entry)==Ok);
    const auto exit=g.journal.prepared(entry,111,0,1,65);
    assert(g.engine.prepared(CorrectReference,exit)==Budget && !g.engine.state().owned_workspace_bytes);
    assert(g.engine.stop(g.config,g.a)==Ok && !g.engine.drain(g.config,g.a));
}
void alias_allocation_and_descriptor_are_quarantined() {
    for(bool workspace:{false,true}) {
        Fixture f;f.qualify(f.a);f.seal({f.a});const auto raw_before=f.journal.state().native_prepares;
        if(workspace)assert(f.admit(Fault,f.a)==Ok);
        f.ops.alias_next_allocation=f.input.data();f.ops.actions.clear();
        if(!workspace) {
            assert(f.admit(Fault,f.a)==Allocation);
            assert(f.journal.state().native_prepares==raw_before);
        } else {
            const auto p=f.descriptors(CorrectReference);const auto entry=f.journal.prepare(p.input,p.quant,p.scale,f.config.thread,0);
            assert(f.engine.bind_prepare(CorrectReference,entry)==Ok);
            const auto exit=f.journal.prepared(entry,111,0,1,16);
            assert(f.engine.prepared(CorrectReference,exit)==Allocation);
        }
        assert(f.engine.state().quarantined_bytes && f.engine.stop(f.config,f.a)==Ok);
        assert(!f.engine.drain(f.config,f.a) && f.engine.recover(f.config)==Pending && !has_release(f.ops));
    }
    Fixture f;f.qualify(f.a);f.seal({f.a});assert(f.admit(Fault,f.a)==Ok);
    assert(f.engine.descriptor(ModelEffect,0,f.call.input.descriptor)==Identity);
    assert(!f.engine.state().owned_descriptors);f.stop_drain({f.a});assert(f.engine.recover(f.config)==Ok);
}
void address_reuse_without_lifetime_proof_cannot_restore() {
    Fixture f;f.qualify(f.a);f.seal({f.a});assert(f.admit(Fault,f.a)==Ok);
    auto ref=f.prepare(CorrectReference),effect=f.prepare(ModelEffect);assert(f.engine.start(71)==Ok);
    f.execute(CorrectReference,ref,71,false);assert(f.engine.readback(CorrectReference)==Ok);
    f.execute(ModelEffect,effect,71,true);assert(f.engine.readback(ModelEffect)==Ok && f.engine.publish()==Ok);
    f.stop_drain({f.a});auto live=f.witness();++live.lifetime_token;f.ops.actions.clear();
    assert(f.engine.recover(f.config,&live)==Liveness && !has_release(f.ops));
    live=f.witness();assert(f.engine.recover(f.config,&live)==Ok);
}
void execute_failure_keeps_unproven_executor_ownership() {
    Fixture f;f.qualify(f.a);f.seal({f.a});assert(f.admit(Fault,f.a)==Ok);
    auto ref=f.prepare(CorrectReference),effect=f.prepare(ModelEffect);assert(f.engine.start(71)==Ok);
    f.execute(CorrectReference,ref,71,false);assert(f.engine.readback(CorrectReference)==Ok);
    f.execute(ModelEffect,effect,71,true,37);assert(f.journal.state().pending==1);
    assert(!f.engine.receipt().effect.drained && !f.engine.receipt().published && f.engine.publish()==StateError);
    assert(f.engine.stop(f.config,f.a)==Ok);f.ops.sync_failure=71;f.ops.actions.clear();
    assert(!f.engine.drain(f.config,f.a) && f.engine.recover(f.config)==Pending && !has_release(f.ops));
    f.ops.sync_failure=0;
    assert(!f.engine.drain(f.config,f.a) && f.engine.recover(f.config)==Pending && !has_release(f.ops));
    assert(f.engine.state().device_bytes && f.journal.state().pending==1);
}
void observer_capacity_duplicate_helper_and_wrong_thread_refuse() {
    Fixture f;f.qualify(f.a);f.seal({f.a});assert(f.admit(Fault,f.a)==Ok);
    const auto p=f.descriptors(CorrectReference);
    const auto raw=f.journal.prepare(p.input,p.quant,p.scale,f.config.thread+1,0);
    assert(f.engine.bind_prepare(CorrectReference,raw)==Identity);
    assert(f.journal.state().native_prepares && f.journal.state().unclassified_prepares==1);
    assert(f.engine.stop(f.config,f.a)==Ok && !f.engine.drain(f.config,f.a));

    Fixture g;g.qualify(g.a);g.seal({g.a});assert(g.admit(Fault,g.a)==Ok);g.prepare(CorrectReference);
    const auto duplicate=g.engine.plan(CorrectReference);
    const auto entry=g.journal.prepare(duplicate.input,duplicate.quant,duplicate.scale,g.config.thread,0);
    assert(g.engine.bind_prepare(CorrectReference,entry)==Identity);
    assert(g.journal.state().helper_prepares==1 && g.journal.state().unclassified_prepares==1);
    assert(g.engine.stop(g.config,g.a)==Ok && !g.engine.drain(g.config,g.a));

    Journal bounded(1);Call c{};assert(bounded.parent(c).token);assert(!bounded.prepare({}, {}, {},41,0));
    bounded.prepared(0,999,0,1);bounded.execute(999,41,0,71);bounded.executed(0,0,1);
    assert(bounded.state().overflow==4 && bounded.state().errors && bounded.state().native_executes==1);
}
void stale_execute_ticket_is_not_a_reused_executor_lease() {
    Fixture f;f.finish(Reference,f.a);const auto previous=f.engine.receipt().effect;
    assert(f.admit(Lossless,f.a)==Ok);f.prepare(ModelEffect,previous.executor);
    assert(f.engine.start(71)==Ok);
    assert(f.engine.bind_execute(ModelEffect,previous.execute_enter)==Identity);
    assert(f.engine.receipt().effect.registration!=previous.registration && !f.engine.receipt().effect.execute_enter);
    assert(f.engine.stop(f.config,f.a)==Ok);f.ops.actions.clear();
    assert(!f.engine.drain(f.config,f.a) && f.engine.recover(f.config)==Pending && !has_release(f.ops));
}
void empty_owner_cancellation_drains_raw_streams_and_retains_failed_owner() {
    Fixture f;const auto p=f.observe();
    auto raw=f.journal.prepare(p.input,p.quant,p.scale,f.config.thread,0);
    f.journal.prepared(raw,777,0,1);
    assert(f.engine.cancel_empty(f.config)==Pending && f.engine.state().installed && f.engine.state().stopped);
    assert(f.admit(Reference,f.a)==Busy);
    auto execute=f.journal.execute(777,f.config.thread,0,71);f.journal.executed(execute,0,1);
    assert(!f.journal.state().pending);f.ops.sync_failure=71;f.ops.actions.clear();
    assert(f.engine.cancel_empty(f.config)==Pending && f.engine.state().installed && !has_release(f.ops));
    assert(f.ops.actions==std::vector<std::string>{"sync:71"});
    f.ops.sync_failure=0;assert(f.engine.cancel_empty(f.config)==Ok && !f.engine.state().installed);
    assert(f.journal.state().native_executes==1 && f.journal.state().unclassified_prepares==1);
    assert(f.engine.open(f.config)==Identity);++f.config.generation;
    assert(f.engine.open(f.config)==Ok);f.ops.actions.clear();
    assert(f.engine.cancel_empty(f.config)==Ok && f.ops.actions.empty());
}
void empty_cancel_cannot_erase_targets_errors_or_foreign_owner() {
    Fixture f;auto wrong=f.config;++wrong.generation;
    assert(f.engine.cancel_empty(wrong)==Identity && !f.engine.state().stopped);
    f.finish(Reference,f.a);const auto host=f.engine.state().host_bytes;
    assert(f.engine.cancel_empty(f.config)==Identity && f.engine.state().installed && f.engine.state().host_bytes==host);
    Fixture g;g.journal.mark_error();
    assert(g.engine.cancel_empty(g.config)==Pending && g.engine.state().installed && g.engine.state().stopped);
}
}
int main(){
    complete_two_targets_actual_reference_effect();missing_helper_wrong_identity_and_no_early_publish();
    wrong_child_role_keeps_raw_census();prepared_failure_with_executor_never_drains();
    no_effect_and_last_input_byte_are_not_hits();stream_failure_and_publish_failure_retain_retry();
    whole_group_barrier_and_complete_qualification();executor_reuse_owner_generation_and_budget();
    unbound_and_missing_callbacks_never_release();old_raw_ticket_cannot_attach_to_new_parent();
    repeated_reference_cannot_replace_frozen_qualification();stopped_ack_waits_for_all_target_drains();
    workspace_has_exact_call_binding_and_ownership();workspace_budget_alias_and_wrong_pointer_refuse();
    alias_allocation_and_descriptor_are_quarantined();address_reuse_without_lifetime_proof_cannot_restore();
    execute_failure_keeps_unproven_executor_ownership();observer_capacity_duplicate_helper_and_wrong_thread_refuse();
    stale_execute_ticket_is_not_a_reused_executor_lease();
    empty_owner_cancellation_drains_raw_streams_and_retains_failed_owner();empty_cancel_cannot_erase_targets_errors_or_foreign_owner();
    std::cout<<"PASS explicit CPU group-core contracts; actual NPU calls=0; no numerical oracle\n";
}
