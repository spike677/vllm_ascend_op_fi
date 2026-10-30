#ifndef NES_FA_PLAN_H
#define NES_FA_PLAN_H
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"

namespace nesfa {
using Json = nlohmann::json;
inline constexpr const char* kSourceCommit = "c77c0829f1f0f0b8f882db1d9b051a758b605472";
inline constexpr const char* kCannBuild = "V100R001C11SPC001B243";
inline constexpr const char* kBridgeAbi = "nes.fa.cann9.1.s1s2.v1";
inline void Require(bool ok, const char* message) { if(!ok)throw std::runtime_error(message); }
inline bool Token(const std::string& s) { return !s.empty() && s.size()<=128 && std::all_of(s.begin(),s.end(),[](char c){return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='.'||c=='-';}); }
inline bool Hex(const std::string& s) { return s.size()==64 && std::all_of(s.begin(),s.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');}); }
inline void Keys(const Json& j, std::initializer_list<const char*> names) {
    Require(j.is_object() && j.size()==names.size(),"Unexpected plan object keys");
    for(const auto* name:names)Require(j.contains(name),"Missing plan key");
}
inline int64_t Integer(const Json& j) {
    Require(j.is_number_integer(),"Plan requires integer");
    if(j.is_number_unsigned())Require(j.get<uint64_t>()<=static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),"Integer overflow");
    return j.get<int64_t>();
}
inline std::string String(const Json& j) { Require(j.is_string(),"Plan requires string");return j.get<std::string>(); }
inline void Shape(const Json& j, std::initializer_list<int64_t> shape) {
    Require(j.is_array() && j.size()==shape.size(),"Plan shape rank");size_t i=0;
    for(auto d:shape)Require(Integer(j[i++])==d,"Plan shape differs from explicit context");
}
struct Factors { int64_t m=0, r=0; };
inline bool Equal(Factors a,Factors b) {return a.m==b.m && a.r==b.r;}
inline bool LegalCandidate(Factors f) {return (f.m==64||f.m==128)&&(f.r==4||f.r==8);}
inline Factors ReadFactors(const Json& j) {
    Keys(j,{"m","n_ratio"});Factors f{Integer(j["m"]),Integer(j["n_ratio"])};
    Require(f.m>0 && f.r>0 && f.m<=2147483647 && f.r<=2147483647,"Nonpositive/oversized factors");return f;
}
inline Json FactorsJson(Factors f) {return Json{{"m",f.m},{"n_ratio",f.r}};}
struct Plan {
    Json document;
    std::string mode, token;
    int64_t pid=0, tid=0, requestCount=0;
    bool expectedNative=false;
    Factors native, selected;
};
inline Plan ParsePlan(const std::string& text) {
    Require(!text.empty() && text.size()<=16384,"Missing/oversized typed plan");
    // Reject duplicate object keys instead of silently accepting last-wins JSON.
    std::vector<std::set<std::string>> objectKeys;
    auto callback=[&](int,Json::parse_event_t event,Json& parsed) {
        if(event==Json::parse_event_t::object_start)objectKeys.emplace_back();
        else if(event==Json::parse_event_t::key){Require(!objectKeys.empty(),"JSON key outside object");Require(objectKeys.back().insert(parsed.get<std::string>()).second,"Duplicate JSON key");}
        else if(event==Json::parse_event_t::object_end){Require(!objectKeys.empty(),"JSON object mismatch");objectKeys.pop_back();}
        return true;
    };
    Plan p;p.document=Json::parse(text,callback);const auto& j=p.document;
    Keys(j,{"schema","operator","mode","scope","context","version","requested","certificate"});
    Require(String(j["schema"])=="nes.fa.typed_plan.v1" && String(j["operator"])=="FlashAttentionScore","Plan schema/operator");
    p.mode=String(j["mode"]);Require(p.mode=="native"||p.mode=="controlled","Plan mode");
    const auto& s=j["scope"];Keys(s,{"pid","native_tid","token","phase","request_count"});
    p.pid=Integer(s["pid"]);p.tid=Integer(s["native_tid"]);p.requestCount=Integer(s["request_count"]);p.token=String(s["token"]);
    Require(p.pid>0 && p.tid>0 && p.requestCount>0 && p.requestCount<=2147483647 && Token(p.token),"Invalid scope identity");
    Require(String(s["phase"])=="operator_execution","Scope phase");
    const auto& v=j["version"];Keys(v,{"source_commit","cann_build","bridge_abi"});
    Require(String(v["source_commit"])==kSourceCommit && String(v["cann_build"])==kCannBuild && String(v["bridge_abi"])==kBridgeAbi,"Unmatched source/build ABI");
    const auto& c=j["context"];Keys(c,{"query_shape","key_shape","value_shape","mask_shape","dtype","mask_dtype","layout","scale","keep_prob","pre_tokens","next_tokens","heads","inner_precise","sparse_mode","optional_inputs_absent"});
    Shape(c["query_shape"],{p.requestCount,16,2048,128});Shape(c["key_shape"],{p.requestCount,8,2048,128});
    Shape(c["value_shape"],{p.requestCount,8,2048,128});Shape(c["mask_shape"],{2048,2048});
    Require(String(c["dtype"])=="float16"&&String(c["mask_dtype"])=="bool"&&String(c["layout"])=="BNSD","Plan dtype/layout");
    Require(c["scale"].is_number() && c["scale"].get<double>()==static_cast<double>(static_cast<float>(1.0/std::sqrt(128.0))),"Plan scale");
    Require(c["keep_prob"].is_number()&&c["keep_prob"].get<double>()==1.0,"Plan keep probability");
    Require(Integer(c["pre_tokens"])==2147483647&&Integer(c["next_tokens"])==0&&Integer(c["heads"])==16&&Integer(c["inner_precise"])==0&&Integer(c["sparse_mode"])==2,"Plan scalar precision/mask attrs");
    Require(c["optional_inputs_absent"].is_boolean()&&c["optional_inputs_absent"].get<bool>(),"Plan optional inputs");
    const auto& cert=j["certificate"];Keys(cert,{"id","source_manifest_sha256","build_manifest_sha256","validation_sha256","validation_state"});
    Require(Token(String(cert["id"]))&&Hex(String(cert["source_manifest_sha256"]))&&Hex(String(cert["build_manifest_sha256"]))&&Hex(String(cert["validation_sha256"])),"Certificate shape");
    const auto cs=String(cert["validation_state"]);Require(cs=="experimental"||cs=="validated","Certificate validation state");
    const auto& req=j["requested"];Keys(req,{"native","selected"});
    p.expectedNative=!req["native"].is_null();if(p.expectedNative)p.native=ReadFactors(req["native"]);
    if(p.mode=="native")Require(req["selected"].is_null(),"Native mode cannot request override");
    else {
        Require(p.requestCount==1||p.requestCount==8,"Controlled batch not qualified");
        Require(p.expectedNative&&!req["selected"].is_null(),"Controlled needs both native/selected factors");
        p.selected=ReadFactors(req["selected"]);
        Require(LegalCandidate(p.native)&&LegalCandidate(p.selected)&&!Equal(p.native,p.selected),"Controlled candidate not qualified or unchanged");
    }
    return p;
}
struct FactorDecision {Factors native, selected;bool applied=false;std::string reason;};
inline FactorDecision DecideFactors(const Plan& p,Factors actual,bool contextEligible) {
    FactorDecision d{actual,actual,false,"context_rejected"};
    if(!contextEligible)return d;
    if(p.expectedNative&&!Equal(p.native,actual)){d.reason="native_factors_mismatch";return d;}
    if(p.mode=="native"){d.reason="native_observed_passthrough";return d;}
    // Both values are returned together, after all validation. No partial override.
    d.selected=p.selected;d.applied=true;d.reason="controlled_pair_selected";return d;
}
} // namespace nesfa
#endif
