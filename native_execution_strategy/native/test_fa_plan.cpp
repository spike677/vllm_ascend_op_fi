#include "fa_plan.h"
#include <iostream>
#include <functional>
using namespace nesfa;

Json Fixture(int64_t b=1) {
    return {{"schema","nes.fa.typed_plan.v1"},{"operator","FlashAttentionScore"},{"mode","native"},
        {"scope",{{"pid",123},{"native_tid",123},{"token","cpu_fixture"},{"phase","operator_execution"},{"request_count",b}}},
        {"context",{{"query_shape",{b,16,2048,128}},{"key_shape",{b,8,2048,128}},{"value_shape",{b,8,2048,128}},
            {"mask_shape",{2048,2048}},{"dtype","float16"},{"mask_dtype","bool"},{"layout","BNSD"},
            {"scale",static_cast<double>(static_cast<float>(1.0/std::sqrt(128.0)))},{"keep_prob",1},
            {"pre_tokens",2147483647},{"next_tokens",0},{"heads",16},{"inner_precise",0},{"sparse_mode",2},{"optional_inputs_absent",true}}},
        {"version",{{"source_commit",kSourceCommit},{"cann_build",kCannBuild},{"bridge_abi",kBridgeAbi}}},
        {"requested",{{"native",nullptr},{"selected",nullptr}}},
        {"certificate",{{"id","synthetic_cpu_fixture_not_a_certificate"},{"source_manifest_sha256",std::string(64,'0')},
            {"build_manifest_sha256",std::string(64,'1')},{"validation_sha256",std::string(64,'2')},{"validation_state","experimental"}}}};
}

int main() {
    try {
        int native=0,controlled=0,negative=0;
        for(auto b:{1,2,3,4,8,16}) {
            auto plan=ParsePlan(Fixture(b).dump());auto d=DecideFactors(plan,{128,8},true);
            Require(!d.applied&&Equal(d.native,d.selected),"Native must preserve factors");++native;
        }
        for(auto b:{1,8})for(Factors f:std::vector<Factors>{{64,8},{128,4},{64,4}}) {
            auto j=Fixture(b);j["mode"]="controlled";j["requested"]={{"native",FactorsJson({128,8})},{"selected",FactorsJson(f)}};
            auto p=ParsePlan(j.dump());
            for(int i=0;i<3;++i){auto d=DecideFactors(p,{128,8},true);Require(d.applied&&Equal(d.selected,f),"Every matched invocation must select");++controlled;}
            auto mismatched=DecideFactors(p,{64,4},true);Require(!mismatched.applied&&Equal(mismatched.selected,{64,4}),"Native mismatch must be atomic passthrough");
            auto rejected=DecideFactors(p,{128,8},false);Require(!rejected.applied&&Equal(rejected.selected,{128,8}),"Context rejection must be atomic passthrough");
        }
        const std::vector<std::function<void(Json&)>> invalid={
            [](Json&j){j["schema"]="wrong";},[](Json&j){j["operator"]="FIA";},[](Json&j){j["mode"]="batch_m";},
            [](Json&j){j["scope"]["request_count"]=1.0;},[](Json&j){j["scope"]["request_count"]=true;},
            [](Json&j){j["scope"]["request_count"]=0;},[](Json&j){j["scope"]["phase"]="operator_replay";},
            [](Json&j){j["scope"]["token"]="bad token";},[](Json&j){j["scope"]["pid"]=0;},
            [](Json&j){j["version"]["source_commit"]="master";},[](Json&j){j["version"]["bridge_abi"]="future";},
            [](Json&j){j["context"]["query_shape"][0]=8;},[](Json&j){j["context"]["dtype"]="bfloat16";},
            [](Json&j){j["context"]["sparse_mode"]=0;},[](Json&j){j["context"]["scale"]=0.1;},
            [](Json&j){j["context"]["optional_inputs_absent"]=false;},[](Json&j){j["context"]["inner_precise"]=2;},
            [](Json&j){j["certificate"]["validation_state"]="assumed_valid";},[](Json&j){j["certificate"]["build_manifest_sha256"]="bad";},
            [](Json&j){j["requested"]["selected"]=FactorsJson({64,4});},
            [](Json&j){j["mode"]="controlled";j["requested"]={{"native",FactorsJson({128,8})},{"selected",FactorsJson({128,8})}};},
            [](Json&j){j=Fixture(2);j["mode"]="controlled";j["requested"]={{"native",FactorsJson({128,8})},{"selected",FactorsJson({64,4})}};},
            [](Json&j){j=Fixture(4);j["mode"]="controlled";j["requested"]={{"native",FactorsJson({128,8})},{"selected",FactorsJson({64,4})}};},
            [](Json&j){j["mode"]="controlled";j["requested"]={{"native",FactorsJson({128,8})},{"selected",FactorsJson({32,4})}};},
            [](Json&j){j["unexpected"]=1;}};
        for(const auto& alter:invalid){auto j=Fixture();alter(j);bool rejected=false;try{ParsePlan(j.dump());}catch(...){rejected=true;}Require(rejected,"Invalid plan accepted");++negative;}
        for(const auto& text:std::vector<std::string>{"{}",std::string(16385,'x'),"{\"mode\":\"native\",\"mode\":\"controlled\"}"}){
            bool rejected=false;try{ParsePlan(text);}catch(...){rejected=true;}Require(rejected,"Bad JSON accepted");++negative;
        }
        std::cout<<Json{{"status","PASS_CPU_TYPED_PLAN_AND_ATOMIC_FACTORS"},{"native_passthroughs",native},
            {"matched_controlled_invocations",controlled},{"invalid_plan_rejections",negative},
            {"npu_executed",false},{"certificate_authenticity_verified",false},{"full_gert_context_tested",false},
            {"new_binary_runtime_validated",false}}.dump()<<'\n';
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
