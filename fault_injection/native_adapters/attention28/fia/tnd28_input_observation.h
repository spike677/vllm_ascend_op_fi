#ifndef DEMONCAT_TND28_INPUT_OBSERVATION_H
#define DEMONCAT_TND28_INPUT_OBSERVATION_H
// Included only in the source-pinned, freshly compiled FIA inner API unit.
// No symbol interposition, extra prepare/execute, device read, or F permission.
#ifndef DCAT_TND28_INPUT_CPU_FIXTURE
#include "opdev/common_types.h"
#include "opdev/op_executor.h"
#include "tnd28_execute_borrow.h"
#include <memory>
#endif
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <iomanip>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <time.h>
#include <unistd.h>

extern "C" int dcat_attention_observer_scope_matches_v1(const char*) __attribute__((weak));
namespace tnd28_input {
inline uint64_t Ptr(const void* p){return reinterpret_cast<uint64_t>(p);}
inline uint64_t Now(){timespec t{};if(clock_gettime(CLOCK_MONOTONIC,&t))throw std::runtime_error("clock");return uint64_t(t.tv_sec)*1000000000ULL+t.tv_nsec;}
inline std::string Env(const char* n){const auto p=std::getenv(n);return p?p:"";}
inline bool Hex(const std::string& s,size_t n){return s.size()==n && s.find_first_not_of("0123456789abcdef")==std::string::npos;}
inline bool Decimal(const std::string& s){return !s.empty() && s.size()<=20 && s[0]!='0' && s.find_first_not_of("0123456789")==std::string::npos;}
inline void Need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
inline std::string Quote(const std::string& s){std::ostringstream o;o<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')o<<'\\'<<c;else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;else o<<c;}o<<'"';return o.str();}
inline std::string ProcessStart(){std::ifstream f("/proc/self/stat");std::string line,word;Need(bool(std::getline(f,line)),"process stat read");const auto end=line.rfind(')');Need(end!=std::string::npos,"process stat format");std::istringstream s(line.substr(end+1));for(unsigned i=3;i<=22;++i)Need(bool(s>>word),"process stat field");return word;}
template<class V> std::string Array(const V& v){Need(v.size()<=64,"array bound");std::ostringstream o;o<<'[';size_t i=0;for(const auto& item:v){if(i++)o<<',';o<<item;}o<<']';return o.str();}
inline std::string Tensor(const aclTensor* t){
    if(!t)return "null";
    std::vector<int64_t> shape,storage,strides;
    const auto& a=t->GetViewShape();const auto& b=t->GetStorageShape();const auto& c=t->GetViewStrides();
    Need(a.GetDimNum()<=8 && b.GetDimNum()<=8 && c.size()<=8,"tensor rank");
    for(size_t i=0;i<a.GetDimNum();++i)shape.push_back(a.GetDim(i));
    for(size_t i=0;i<b.GetDimNum();++i)storage.push_back(b.GetDim(i));
    for(size_t i=0;i<c.size();++i)strides.push_back(c[i]);
    std::ostringstream o;o<<"{\"descriptor\":"<<Ptr(t)<<",\"storage_address\":"<<Ptr(t->GetStorageAddr())
      <<",\"data_address\":"<<Ptr(t->GetData())<<",\"offset\":"<<t->GetViewOffset()
      <<",\"native_storage_offset\":"<<t->GetStorageOffset()
      <<",\"dtype\":"<<int(t->GetDataType())<<",\"format\":"<<int(t->GetStorageFormat())
      <<",\"shape\":"<<Array(shape)<<",\"stride\":"<<Array(strides)<<",\"storage_shape\":"<<Array(storage)<<'}';return o.str();
}
inline std::string Tensors(const aclTensorList* list){
    if(!list)return "null";Need(list->Size()<=8,"tensor list bound");std::ostringstream o;
    o<<"{\"object\":"<<Ptr(list)<<",\"members\":[";
    for(size_t i=0;i<list->Size();++i){if(i)o<<',';o<<Tensor((*list)[i]);}o<<"]}";return o.str();
}
inline std::string Integers(const aclIntArray* list){
    if(!list)return "null";Need(list->Size()<=8,"integer list bound");std::vector<int64_t> v;
    for(size_t i=0;i<list->Size();++i)v.push_back((*list)[i]);
    return "{\"object\":"+std::to_string(Ptr(list))+",\"values\":"+Array(v)+'}';
}
template<class F> std::string Function(F f){
    const auto p=reinterpret_cast<const void*>(f);Dl_info d{};
    Need(p && dladdr(p,&d) && d.dli_fbase && d.dli_fname,"actual function owner unavailable");
    uint64_t begin=0,end=0,inode=0,deviceMajor=0,deviceMinor=0;std::string permissions,mappedPath;
    std::ifstream maps("/proc/self/maps");std::string line;size_t lines=0;
    while(std::getline(maps,line)){
        Need(++lines<=65536&&line.size()<=8192,"maps bound");std::istringstream in(line);
        std::string range,offset,dev;uint64_t ino=0;in>>range>>permissions>>offset>>dev>>ino;
        const auto dash=range.find('-'),colon=dev.find(':');if(dash==std::string::npos||colon==std::string::npos)continue;
        const auto lo=std::stoull(range.substr(0,dash),nullptr,16),hi=std::stoull(range.substr(dash+1),nullptr,16);
        if(Ptr(p)<lo||Ptr(p)>=hi)continue;
        begin=lo;end=hi;inode=ino;deviceMajor=std::stoull(dev.substr(0,colon),nullptr,16);deviceMinor=std::stoull(dev.substr(colon+1),nullptr,16);
        std::getline(in,mappedPath);const auto nonspace=mappedPath.find_first_not_of(' ');mappedPath=nonspace==std::string::npos?"":mappedPath.substr(nonspace);break;
    }
    struct stat st{};const bool exists=stat(d.dli_fname,&st)==0;
    const bool same=exists&&begin&&permissions.find('x')!=std::string::npos&&inode==uint64_t(st.st_ino)
        &&deviceMajor==major(st.st_dev)&&deviceMinor==minor(st.st_dev)&&S_ISREG(st.st_mode);
    return "{\"address\":"+std::to_string(Ptr(p))+",\"base\":"+std::to_string(Ptr(d.dli_fbase))+
      ",\"path\":"+Quote(d.dli_fname)+",\"symbol\":"+Quote(d.dli_sname?d.dli_sname:"")+
      ",\"mapped_path\":"+Quote(mappedPath)+",\"mapped_begin\":"+std::to_string(begin)+",\"mapped_end\":"+std::to_string(end)+
      ",\"mapped_inode\":"+std::to_string(inode)+",\"mapped_device\":["+std::to_string(deviceMajor)+','+std::to_string(deviceMinor)+"]"+
      ",\"file_inode\":"+std::to_string(exists?st.st_ino:0)+",\"file_bytes\":"+std::to_string(exists?st.st_size:0)+
      ",\"file_device\":["+std::to_string(exists?major(st.st_dev):0)+','+std::to_string(exists?minor(st.st_dev):0)+"]"+
      ",\"mapped_file_identity_matches\":"+(same?"true":"false")+'}';
}
struct Scope{std::string token,owner,generation,module,request,input,path,start;uint64_t pid=0,tid=0;};
inline bool Current(Scope& s){
    if(Env("DCAT28_INPUT_MODE")!="observe")return false;
    s.token=Env("DCAT28_INPUT_TOKEN");s.pid=getpid();s.tid=syscall(SYS_gettid);
    if(!Hex(s.token,32) || Env("DCAT28_INPUT_PID")!=std::to_string(s.pid)
      || Env("DCAT28_INPUT_TID")!=std::to_string(s.tid) || !dcat_attention_observer_scope_matches_v1
      || dcat_attention_observer_scope_matches_v1(s.token.c_str())!=1)return false;
    s.owner=Env("DCAT28_INPUT_OWNER");s.generation=Env("DCAT28_INPUT_GENERATION");s.module=Env("DCAT28_INPUT_MODULE");
    s.request=Env("DCAT28_INPUT_REQUEST");s.input=Env("DCAT28_INPUT_HASH");s.path=Env("DCAT28_INPUT_LOG");s.start=Env("DCAT28_INPUT_START");
    Need(Hex(s.owner,32)&&Decimal(s.generation)&&Decimal(s.module)&&Hex(s.request,64)&&Hex(s.input,64)&&Decimal(s.start),"input owner identity");
    Need(s.start==ProcessStart(),"input process lifetime changed");
    Need(s.path.rfind("/evidence/",0)==0&&s.path.size()<4096&&s.path.find("/../")==std::string::npos
      &&s.path.find("/./")==std::string::npos&&s.path.find_first_of("\r\n")==std::string::npos,"input log path");
    return true;
}
inline bool Same(const Scope& a,const Scope& b){return a.token==b.token&&a.owner==b.owner&&a.generation==b.generation
    &&a.module==b.module&&a.request==b.request&&a.input==b.input&&a.path==b.path&&a.start==b.start&&a.pid==b.pid&&a.tid==b.tid;}
// A trusted live Python call object checks these actual generated-unit values.
// This is a normal guard, not the host tiling/F authority (including v1).
using PublicCheck = int (*)(const char*, uint32_t, const char*, size_t);
struct PublicPlan {
    Scope scope; PublicCheck callback=nullptr; std::string typed;
    bool checking=false,prepare_entered=false,prepare_accepted=false,prepare_admitted=false,execute_entered=false;
    bool safe_release=false,terminal_uncertain=false; uint32_t first_error=0;
};
inline PublicPlan& Plan(){static thread_local PublicPlan value;return value;}
inline bool PlanOwner(const PublicPlan& plan,const char* token){
    return token&&std::strlen(token)==32&&plan.scope.token==token
      &&plan.scope.pid==static_cast<uint64_t>(getpid())&&plan.scope.tid==static_cast<uint64_t>(syscall(SYS_gettid))
      &&plan.scope.start==ProcessStart();
}
inline std::string PlanRecord(){const auto& p=Plan();
    return "{\"schema\":\"demoncat.tnd28.live_typed_plan.v2\",\"bound\":"+std::to_string(p.callback!=nullptr)
      +",\"prepare_entered\":"+std::to_string(p.prepare_entered)+",\"prepare_accepted\":"+std::to_string(p.prepare_accepted)
      +",\"prepare_admitted\":"+std::to_string(p.prepare_admitted)
      +",\"execute_entered\":"+std::to_string(p.execute_entered)+",\"safe_release\":"+std::to_string(p.safe_release)
      +",\"terminal_uncertain\":"+std::to_string(p.terminal_uncertain)+",\"first_error\":"+std::to_string(p.first_error)
      +",\"fault_authorized\":false}";
}
inline bool CheckPublic(const Scope& scope,uint32_t stage,const std::string& body){
    auto& p=Plan();Scope current;
    if(!p.callback||p.checking||!Same(p.scope,scope)||!Current(current)||!Same(current,scope)
       ||p.first_error||body.empty()||body.size()>32768){if(!p.first_error)p.first_error=1;return false;}
    if(stage==1){
        if(p.prepare_entered){p.first_error=2;return false;}
        p.prepare_entered=true;p.typed=body;
    }else if(stage==2){
        if(!p.prepare_accepted||!p.prepare_admitted||p.execute_entered){p.first_error=2;return false;}
        p.execute_entered=true;
    }else{p.first_error=2;return false;}
    p.checking=true;int status=0;
    try{status=p.callback(scope.token.c_str(),stage,body.c_str(),body.size());}
    catch(...){p.checking=false;if(!p.first_error)p.first_error=3;return false;}
    p.checking=false;
    if(status!=1){if(!p.first_error)p.first_error=3;if(stage==1)p.safe_release=true;return false;}
    if(stage==1)p.prepare_accepted=true;
    return true;
}
struct Pending{Scope scope;uint64_t executor=0,child=0,child_bytes=0;std::vector<uint64_t> offsets;bool valid=false;};
inline Pending& Live(){static thread_local Pending value;return value;}
#ifndef DCAT_TND28_INPUT_CPU_FIXTURE
struct BorrowReturn {Scope scope;std::string receipt;};
inline BorrowReturn& LastBorrowReturn(){static thread_local BorrowReturn value;return value;}
#endif
inline std::set<std::string>& Seen(){static thread_local std::set<std::string> value;return value;}
struct Log{
    Scope scope;bool active=false;uint32_t count=0;std::string first_error;
    void Fail(const std::string& reason) noexcept {try{if(first_error.empty())first_error=reason;}catch(...) {}}
    template<class F> void Record(const char* phase,F body) noexcept {
        if(!active)return;
        try{
            Scope current;Need(Current(current)&&Same(scope,current),"current input scope changed");Need(++count<=24,"event budget");
            std::ostringstream o;o<<std::setprecision(17)<<"{\"schema\":\"demoncat.tnd28.compiled_input.v1\",\"phase\":"<<Quote(phase)
              <<",\"token\":"<<Quote(scope.token)<<",\"owner\":"<<Quote(scope.owner)<<",\"generation\":"<<scope.generation
              <<",\"module_object_id\":"<<scope.module<<",\"request_sha256\":"<<Quote(scope.request)
              <<",\"input_sha256\":"<<Quote(scope.input)<<",\"process_start_ticks\":"<<Quote(scope.start)
              <<",\"pid\":"<<scope.pid<<",\"tid\":"<<scope.tid<<",\"monotonic_ns\":"<<Now()
              <<",\"first_error\":"<<Quote(first_error)<<",\"fault_authorized\":false,\"data\":"<<body()<<"}\n";
            const auto text=o.str();Need(text.size()<=65536,"event byte bound");
#ifdef DCAT_TND28_INPUT_CPU_FIXTURE
            tnd28_test_lines.push_back(text);
#else
            const int fd=open(scope.path.c_str(),O_WRONLY|O_APPEND|O_CREAT|O_NOFOLLOW|O_CLOEXEC,0600);
            Need(fd>=0,"input log open");struct stat st{};
            if(fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_nlink!=1||st.st_size<0||uint64_t(st.st_size)+text.size()>1048576){close(fd);throw std::runtime_error("input log bound or identity");}
            const auto n=write(fd,text.data(),text.size());const int closed=close(fd);
            Need(n==static_cast<ssize_t>(text.size())&&!closed,"input log write");
#endif
        }catch(const std::exception& e){Fail(e.what());}catch(...){Fail("observer exception");}
    }
};
struct Prepare:Log{
    const aclTensor* value=nullptr;bool ran=false;bool child_ok=false;bool returned=false;
    bool guarded=false,admitted=false;
    Pending pending;
    template<class F> Prepare(F function,const aclTensor* q,const aclTensorList* k,const aclTensorList* v,
      const aclTensor* mask,const aclIntArray* sq,const aclIntArray* sk,const aclTensor* out,
      const aclTensor* lse,std::initializer_list<const void*> optional,std::initializer_list<int64_t> scalars,
      double scale,const char* layout,bool softmax){
        guarded=Env("DCAT28_INPUT_EXECUTE_BORROW")=="normal_guard";
        try{active=Current(scope);if(!active)return;
            Need(!Live().executor&&Seen().size()<4&&Seen().insert(scope.token).second,"pending or repeated input scope");
            pending.scope=scope;
            Record("prepare_enter",[&]{
                Need(v&&v->Size()==1,"single actual V member required");value=(*v)[0];std::vector<uint64_t> pointers;
                for(auto p:optional)pointers.push_back(Ptr(p));
                Need(layout&&std::strlen(layout)<=8,"layout bound");
                const auto body="{\"function\":"+Function(function)+",\"query\":"+Tensor(q)+",\"key\":"+Tensors(k)+",\"value\":"+Tensors(v)
                  +",\"mask\":"+Tensor(mask)+",\"seq_q\":"+Integers(sq)+",\"seq_kv\":"+Integers(sk)
                  +",\"output\":"+Tensor(out)+",\"lse\":"+Tensor(lse)+",\"optionals\":"+Array(pointers)
                  +",\"scalars\":"+Array(scalars)+",\"scale\":"+([&]{std::ostringstream o;o<<std::setprecision(17)<<scale;return o.str();})()
                  +",\"layout\":"+Quote(layout)+",\"softmax_lse\":"+(softmax?"true":"false")+'}';
                if(guarded)admitted=CheckPublic(scope,1,body);
                return body;});
        }catch(const std::exception& e){Fail(e.what());}catch(...){Fail("prepare entry exception");}
    }
    bool Admit(){
        // The generated wrapper calls this before allocating a parent. A
        // failed typed check therefore has no executor/workspace to destroy.
        if(!guarded)return true;
        const bool ok=admitted&&first_error.empty();
        auto& p=Plan();
        if(ok)p.prepare_admitted=true;
        else if(Same(p.scope,scope)){p.safe_release=true;if(!p.first_error)p.first_error=4;}
        return ok;
    }
    template<class F> aclnnStatus Contiguous(F f,const aclTensorList* list,uint64_t* bytes,aclOpExecutor** executor){
        Record("contiguous_prepare_enter",[&]{Need(list&&list->Size()==1&&(*list)[0]==value,"actual noncontiguous list is not original V");
          return "{\"function\":"+Function(f)+",\"inputs\":"+Tensors(list)+'}';});
        const auto status=f(list,bytes,executor); // unchanged original arguments, once
        Record("contiguous_prepare_return",[&]{Need(bytes&&executor,"prepare outputs absent");pending.child=Ptr(*executor);pending.child_bytes=*bytes;
          if(!status&&*executor){const auto& offsets=(*executor)->GetWorkspaceOffsets();Need(offsets.size()<=8,"offset bound");
            pending.offsets.assign(offsets.begin(),offsets.end());child_ok=pending.offsets.size()==1;}
          return "{\"status\":"+std::to_string(status)+",\"original_calls\":1,\"executor\":"+std::to_string(pending.child)
            +",\"workspace_bytes\":"+std::to_string(pending.child_bytes)+",\"workspace_offsets\":"+Array(pending.offsets)+'}';});
        return status;
    }
    template<class F> aclnnStatus Workspace(F f,void* executor,uint64_t* bytes){
        Record("workspace_enter",[&]{return "{\"function\":"+Function(f)+",\"executor\":"+std::to_string(Ptr(executor))+'}';});
        const auto status=f(executor,bytes);ran=true;
        Record("workspace_return",[&]{Need(bytes,"workspace output absent");pending.executor=Ptr(executor);
          pending.valid=!status&&child_ok&&first_error.empty();Live()=pending;
          return "{\"status\":"+std::to_string(status)+",\"original_calls\":1,\"executor\":"+std::to_string(pending.executor)
            +",\"fia_workspace_bytes\":"+std::to_string(*bytes)+",\"typed_copy_descriptor_join\":"+(pending.valid?"true":"false")+'}';});
        return status;
    }
    aclnnStatus Return(aclnnStatus status,uint64_t* bytes,aclOpExecutor** executor){
        returned=true;Record("prepare_return",[&]{Need(bytes&&executor,"original prepare outputs absent");
          return "{\"status\":"+std::to_string(status)+",\"original_returns\":1,\"executor\":"+std::to_string(Ptr(*executor))
              +",\"total_workspace_bytes\":"+std::to_string(*bytes)+'}';});return status;
    }
    ~Prepare(){Record("prepare_exit",[&]{return std::string("{\"workspace_called\":")+(ran?"true":"false")
      +",\"original_returned\":"+(returned?"true":"false")+'}';});}
};
struct Execute:Log{
    Pending pending;uint64_t parent=0,child_workspace=0,child_stream=0;bool child_returned=false,ran=false,returned=false;
#ifndef DCAT_TND28_INPUT_CPU_FIXTURE
    std::unique_ptr<tnd28_execute_borrow::ExecuteBorrow> borrow;
    bool borrow_requested=false;
    std::string BorrowRecord(){
        if(!borrow)return "null";const auto r=borrow->Snapshot();
        return "{\"schema\":\"demoncat.tnd28.execute_borrow.v1\",\"first_error\":"+std::to_string(r.first_error)
          +",\"pid\":"+std::to_string(r.pid)+",\"tid\":"+std::to_string(r.tid)
          +",\"child_started\":"+std::to_string(r.child_started)+",\"child_consumed\":"+std::to_string(r.child_consumed)
          +",\"parent_started\":"+std::to_string(r.parent_started)+",\"parent_returned\":"+std::to_string(r.parent_returned)
          +",\"drained\":"+std::to_string(r.drained)+",\"detached\":"+std::to_string(r.detached)
          +",\"reset_attempted\":"+std::to_string(r.reset_attempted)+",\"reset_returned\":"+std::to_string(r.reset_returned)
          +",\"cancelled\":"+std::to_string(r.cancelled)+",\"complete\":"+std::to_string(r.complete)
          +",\"terminal_uncertain\":"+std::to_string(r.terminal_uncertain)+",\"left\":"+std::to_string(r.left)
          +",\"child_status\":"+std::to_string(r.child_status)+",\"parent_status\":"+std::to_string(r.parent_status)
          +",\"drain_status\":"+std::to_string(r.drain_status)+",\"detach_status\":"+std::to_string(r.detach_status)
          +",\"reset_status\":"+std::to_string(r.reset_status)+",\"fault_authorized\":false}";
    }
#else
    std::string BorrowRecord(){return "null";}
#endif
    template<class F> Execute(F function,aclOpExecutor* executor,void* workspace=nullptr,uint64_t bytes=0,aclrtStream stream=nullptr){
        try{active=Current(scope);if(!active)return;pending=Live();parent=Ptr(executor);
          Record("execute_enter",[&]{Need(pending.executor==parent&&pending.valid&&Same(pending.scope,scope),"current prepared executor unavailable");
            return "{\"function\":"+Function(function)+",\"executor\":"+std::to_string(parent)+'}';});
#ifndef DCAT_TND28_INPUT_CPU_FIXTURE
          borrow_requested=Env("DCAT28_INPUT_EXECUTE_BORROW")=="normal_guard";
          if(borrow_requested){
            Need(first_error.empty(),"borrow requires actual prepared input scope");
            borrow.reset(new tnd28_execute_borrow::ExecuteBorrow(executor,workspace,bytes,stream));
          }
#else
          (void)workspace;(void)bytes;(void)stream;
#endif
        }catch(const std::exception& e){Fail(e.what());}catch(...){Fail("execute entry exception");}
    }
    template<class F> aclnnStatus Contiguous(F f,void* workspace,uint64_t bytes,aclOpExecutor* executor,aclrtStream stream){
        Record("contiguous_execute_enter",[&]{Need(Ptr(executor)==pending.child&&bytes==pending.child_bytes&&workspace,"current child executor differs");
          const auto& offsets=executor->GetWorkspaceOffsets();std::vector<uint64_t> current(offsets.begin(),offsets.end());
          Need(current==pending.offsets,"live workspace offsets changed");child_workspace=Ptr(workspace);child_stream=Ptr(stream);
          Need(current.size()==1&&current[0]<=bytes&&4194304<=bytes-current[0]
            &&child_workspace<=std::numeric_limits<uint64_t>::max()-bytes,"prepared V span outside actual child workspace");
          return "{\"function\":"+Function(f)+",\"executor\":"+std::to_string(Ptr(executor))+",\"workspace\":"+std::to_string(child_workspace)
            +",\"workspace_bytes\":"+std::to_string(bytes)+",\"stream\":"+std::to_string(child_stream)
            +",\"workspace_offsets\":"+Array(current)+",\"prepared_v_address\":"+std::to_string(child_workspace+current[0])+'}';});
        aclnnStatus status;
#ifndef DCAT_TND28_INPUT_CPU_FIXTURE
        if(borrow_requested&&!borrow){Fail("execute borrow not constructed");return ACLNN_ERR_PARAM_INVALID;}
        if(borrow&&(!first_error.empty()||!borrow->BeforeChild(executor,workspace,bytes,stream))){
          borrow->CancelBeforeParent();Fail("execute borrow child admission refused");return ACLNN_ERR_PARAM_INVALID;}
        try{status=f(workspace,bytes,executor,stream);}
        catch(...){if(borrow)borrow->OriginalThrew();throw;}
        if(borrow){borrow->AfterChild(status);if(status)borrow->CancelBeforeParent();}
#else
        status=f(workspace,bytes,executor,stream);
#endif
        Record("contiguous_execute_return",[&]{child_returned=status==0;return "{\"status\":"+std::to_string(status)+",\"original_calls\":1}";});return status;
    }
    template<class F> aclnnStatus Run(F f,void* executor,aclrtStream stream,void* workspace,uint64_t bytes){
        Record("fia_execute_enter",[&]{Need(Ptr(executor)==parent&&child_returned&&Ptr(stream)==child_stream
          &&Ptr(workspace)<=std::numeric_limits<uint64_t>::max()-bytes&&Ptr(workspace)+bytes==child_workspace,"actual parent/child workspace or stream differs");
          return "{\"function\":"+Function(f)+",\"executor\":"+std::to_string(parent)+",\"workspace\":"+std::to_string(Ptr(workspace))
            +",\"workspace_bytes\":"+std::to_string(bytes)+",\"stream\":"+std::to_string(Ptr(stream))+'}';});
        aclnnStatus status;
#ifndef DCAT_TND28_INPUT_CPU_FIXTURE
        if(borrow_requested&&!borrow){Fail("execute borrow not constructed");return ACLNN_ERR_PARAM_INVALID;}
        if(borrow&&(!first_error.empty()||!borrow->PrepareParent(static_cast<aclOpExecutor*>(executor),workspace,bytes,stream))){
          borrow->CancelBeforeParent();Fail("execute borrow parent preparation refused");return ACLNN_ERR_PARAM_INVALID;}
        // Still only the existing normal metadata guard. Draining first enables
        // a future fresh prepared-input check, but does not make saved typed
        // JSON a numerical proof or authorize denominator replay.
        if(borrow&&!CheckPublic(scope,2,Plan().typed)){
          borrow->CancelBeforeParent();Fail("live public input changed before parent execute");return ACLNN_ERR_PARAM_INVALID;}
        if(borrow&&!borrow->CommitParent(static_cast<aclOpExecutor*>(executor),workspace,bytes,stream)){
          borrow->CancelBeforeParent();Fail("execute borrow parent admission refused");return ACLNN_ERR_PARAM_INVALID;}
        try{status=f(executor,stream,workspace,bytes);ran=true;}
        catch(...){if(borrow)borrow->OriginalThrew();throw;}
        const bool borrow_complete=!borrow||borrow->AfterParent(status);
#else
        status=f(executor,stream,workspace,bytes);ran=true;
#endif
        Record("fia_execute_return",[&]{return "{\"status\":"+std::to_string(status)+",\"original_calls\":1}";});
        if(!status&&active&&Live().executor==parent&&Same(Live().scope,scope))Live()=Pending{};
#ifndef DCAT_TND28_INPUT_CPU_FIXTURE
        if(!borrow_complete){Fail("execute borrow drain or parent reset unconfirmed");return status?status:ACLNN_ERR_PARAM_INVALID;}
#endif
        return status;
    }
    aclnnStatus Return(aclnnStatus status){returned=true;
        Record("execute_return",[&]{return "{\"status\":"+std::to_string(status)+",\"original_returns\":1}";});return status;
    }
    ~Execute(){
#ifndef DCAT_TND28_INPUT_CPU_FIXTURE
      if(borrow){borrow->Leave();const auto r=borrow->Snapshot();auto& p=Plan();
        if(p.callback&&Same(p.scope,scope)){p.safe_release=r.complete&&r.drained&&!r.terminal_uncertain;p.terminal_uncertain=r.terminal_uncertain;}
        try{LastBorrowReturn()={scope,BorrowRecord()};}catch(...){Fail("borrow receipt publication failed");}}
      if(borrow_requested&&Live().executor==parent&&Same(Live().scope,scope))Live()=Pending{};
#endif
      Record("execute_exit",[&]{return std::string("{\"fia_called\":")+(ran?"true":"false")
      +",\"original_returned\":"+(returned?"true":"false")+",\"execute_borrow\":"+BorrowRecord()+'}';});}
};
} // namespace tnd28_input
#ifndef DCAT_TND28_INPUT_CPU_FIXTURE
extern "C" __attribute__((visibility("default"))) int dcat_tnd28_public_plan_bind_v2(
    const char* token,tnd28_input::PublicCheck callback) noexcept {
    try{tnd28_input::Scope scope;
      if(!token||!callback||!tnd28_input::Current(scope)||scope.token!=token
        ||tnd28_input::Env("DCAT28_INPUT_EXECUTE_BORROW")!="normal_guard"
        ||tnd28_input::Plan().callback||tnd28_input::Live().executor)return -1;
      tnd28_input::PublicPlan value;value.scope=scope;value.callback=callback;tnd28_input::Plan()=std::move(value);return 0;
    }catch(...){return -2;}
}
extern "C" __attribute__((visibility("default"))) int dcat_tnd28_public_plan_clear_v2(
    const char* token,tnd28_input::PublicCheck callback) noexcept {
    try{auto& p=tnd28_input::Plan();
      if(!tnd28_input::PlanOwner(p,token)||p.callback!=callback||!callback||p.checking
        ||p.terminal_uncertain||(p.prepare_entered&&!p.safe_release))return -1;
      p.callback=nullptr;p.typed.clear();return 0;
    }catch(...){return -2;}
}
extern "C" __attribute__((visibility("default"))) int dcat_tnd28_public_plan_query_v2(
    const char* token,char* output,size_t capacity) noexcept {
    try{const auto& p=tnd28_input::Plan();
      if(!output||capacity<2||capacity>4096||!tnd28_input::PlanOwner(p,token))return -1;
      const auto text=tnd28_input::PlanRecord();if(text.size()+1>capacity)return -2;
      std::memcpy(output,text.c_str(),text.size()+1);return 0;
    }catch(...){return -3;}
}
extern "C" __attribute__((visibility("default"))) int dcat_tnd28_execute_borrow_query_v1(
    const char* token,char* output,size_t capacity) noexcept {
    try{
        const auto& row=tnd28_input::LastBorrowReturn();
        if(!token||!output||capacity>4096||capacity<2||std::strlen(token)!=32||row.scope.token!=token
          ||row.scope.pid!=static_cast<uint64_t>(getpid())||row.scope.tid!=static_cast<uint64_t>(syscall(SYS_gettid))
          ||row.scope.start!=tnd28_input::ProcessStart()||row.receipt.empty())return -1;
        const auto text="{\"token\":"+tnd28_input::Quote(row.scope.token)+",\"owner\":"+tnd28_input::Quote(row.scope.owner)
          +",\"generation\":"+row.scope.generation+",\"module_object_id\":"+row.scope.module
          +",\"request_sha256\":"+tnd28_input::Quote(row.scope.request)+",\"input_sha256\":"+tnd28_input::Quote(row.scope.input)
          +",\"borrow\":"+row.receipt+",\"public_plan\":"+tnd28_input::PlanRecord()
          +",\"function\":"+tnd28_input::Function(&dcat_tnd28_execute_borrow_query_v1)+'}';
        if(text.size()+1>capacity)return -2;
        std::memcpy(output,text.c_str(),text.size()+1);return 0;
    }catch(...){return -3;}
}
#endif
#endif
