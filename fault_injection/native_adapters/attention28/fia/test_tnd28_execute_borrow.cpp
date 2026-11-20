// CPU contract for the actual API-shaped cancellation calls, not Native proof.
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <thread>
using aclnnStatus=int;
using aclrtStream=void*;
constexpr uint64_t K_EXECUTOR_MAGIC_NUMBER=0x1F2E3D4C5B6A;
struct aclOpExecutor {
    bool repeat=false;uint64_t magic=K_EXECUTOR_MAGIC_NUMBER;
    uint64_t GetMagicNumber(){return magic;}
    bool IsRepeatable()const{return repeat;}
};
extern "C" int aclrtSynchronizeStream(aclrtStream);
extern "C" aclnnStatus aclDestroyAclOpExecutor(aclOpExecutor*);
#define DCAT_TND28_BORROW_CPU_FIXTURE
#include "tnd28_execute_borrow.h"
using namespace tnd28_execute_borrow;
static aclOpExecutor parent,child,foreign;
static char workspace[1024];
static aclrtStream stream=reinterpret_cast<void*>(1);
static aclOpExecutor* registered=&child;
static size_t registered_bytes=256;
static bool child_live=true,view=false,drain_error=false,detach_before=false,detach_after=false,reset_error=false,readback_error=false;
static unsigned drains=0,detaches=0,resets=0;
extern "C" aclnnStatus NnopbaseGetUnContExecutor(void* p,aclOpExecutor** c,size_t* n){
    assert(p==&parent);if(readback_error&&resets)return 9;*c=registered;*n=registered_bytes;return 0;
}
extern "C" aclnnStatus NnopbaseGetViewCopyExecutor(void* p,aclOpExecutor** c){assert(p==&parent);*c=view?&foreign:nullptr;return 0;}
extern "C" aclnnStatus NnopbaseSetUnContExecutor(void* p,aclOpExecutor* c,size_t n){
    assert(p==&parent&&!c&&!n);++detaches;if(detach_before)return 7;
    registered=nullptr;registered_bytes=0;return detach_after?7:0;
}
extern "C" int aclrtSynchronizeStream(aclrtStream s){assert(s==stream);++drains;return drain_error?6:0;}
extern "C" aclnnStatus aclDestroyAclOpExecutor(aclOpExecutor* p){
    assert(p==&parent);++resets;
    // A consumed child still attached to the parent would be a double destroy.
    if(registered){assert(child_live);child_live=false;}
    registered=nullptr;registered_bytes=0;return reset_error?8:0;
}
static void Reset(){parent={};child={};registered=&child;registered_bytes=256;child_live=true;
    view=drain_error=detach_before=detach_after=reset_error=readback_error=false;drains=detaches=resets=0;}
static void Copy(ExecuteBorrow& b,int status=0){assert(b.BeforeChild(&child,workspace+768,256,stream));
    child_live=false;assert(b.AfterChild(status)==(status==0));}
static void NoRetry(ExecuteBorrow& b){auto before=drains+detaches+resets;b.Leave();
    assert(!b.CancelBeforeParent());assert(drains+detaches+resets==before);assert(b.Snapshot().left);}
int main(){
    Reset();{ExecuteBorrow b(&parent,workspace,1024,stream);Copy(b);
        assert(b.PrepareParent(&parent,workspace,768,stream));
        assert(drains==1&&!b.Snapshot().parent_started);
        // The live input checker rejects: cancel safely after copy, before FIA.
        assert(b.CancelBeforeParent());assert(resets==1&&detaches==1);
        assert(!b.CommitParent(&parent,workspace,768,stream));NoRetry(b);}
    Reset();{ExecuteBorrow b(&parent,workspace,1024,stream);Copy(b);
        assert(!b.CommitParent(&parent,workspace,768,stream));
        assert(!b.Snapshot().parent_started);assert(b.CancelBeforeParent());NoRetry(b);}
    Reset();{ExecuteBorrow b(&parent,workspace,1024,stream);Copy(b);
        assert(b.PrepareParent(&parent,workspace,768,stream));
        assert(!b.CommitParent(&parent,workspace+1,768,stream));
        assert(!b.Snapshot().parent_started);assert(b.CancelBeforeParent());NoRetry(b);}

    Reset();{ExecuteBorrow b(&parent,workspace,1024,stream);Copy(b);
        assert(b.BeforeParent(&parent,workspace,768,stream));
        registered=nullptr;registered_bytes=0; // CPU original NNOPBASE return seam.
        assert(b.AfterParent(0));b.Leave();auto r=b.Snapshot();
        assert(r.complete&&!r.terminal_uncertain&&!r.first_error&&drains==2&&resets==0);NoRetry(b);}
    Reset();{ExecuteBorrow b(&parent,workspace,1024,stream);
        assert(b.CancelBeforeParent());assert(!child_live&&resets==1&&detaches==0);NoRetry(b);}
    Reset();{ExecuteBorrow b(&parent,workspace,1024,stream);Copy(b);
        assert(b.CancelBeforeParent());assert(detaches==1&&resets==1);NoRetry(b);}
    Reset();{ExecuteBorrow b(&parent,workspace,1024,stream);Copy(b,13);
        assert(b.CancelBeforeParent());assert(b.Snapshot().first_error==CHILD_CALL_FAILED);NoRetry(b);}
    Reset();{ExecuteBorrow b(&parent,workspace,1024,stream);Copy(b);drain_error=true;
        assert(!b.CancelBeforeParent());assert(detaches==0&&resets==0);NoRetry(b);
        assert(b.Snapshot().terminal_uncertain&&b.Snapshot().first_error==DRAIN);}
    Reset();{ExecuteBorrow b(&parent,workspace,1024,stream);Copy(b);detach_before=true;
        assert(!b.CancelBeforeParent());assert(resets==0);NoRetry(b);}
    Reset();{ExecuteBorrow b(&parent,workspace,1024,stream);Copy(b);detach_after=true;
        assert(b.CancelBeforeParent());assert(b.Snapshot().first_error==DETACH&&b.Snapshot().complete);
        assert(detaches==1&&resets==1);NoRetry(b);}
    Reset();{ExecuteBorrow b(&parent,workspace,1024,stream);Copy(b);reset_error=true;
        assert(!b.CancelBeforeParent());assert(b.Snapshot().reset_attempted&&!b.Snapshot().reset_returned);
        assert(!b.CancelBeforeParent());assert(resets==1);NoRetry(b);}
    Reset();{ExecuteBorrow b(&parent,workspace,1024,stream);Copy(b);readback_error=true;
        assert(!b.CancelBeforeParent());assert(b.Snapshot().reset_returned);NoRetry(b);assert(resets==1);}
    Reset();{ExecuteBorrow b(&parent,workspace,1024,stream);
        assert(b.BeforeChild(&child,workspace+768,256,stream));b.OriginalThrew();
        assert(!b.CancelBeforeParent());assert(!resets);NoRetry(b);}
    Reset();{ExecuteBorrow b(&parent,workspace,1024,stream);Copy(b);
        bool permitted=true;std::thread t([&]{permitted=b.BeforeParent(&parent,workspace,768,stream);});t.join();
        assert(!permitted&&!drains);assert(b.CancelBeforeParent());NoRetry(b);}
    Reset();{child.repeat=true;ExecuteBorrow b(&parent,workspace,1024,stream);
        assert(!b.BeforeChild(&child,workspace+768,256,stream));NoRetry(b);assert(!resets&&!drains);}
    Reset();{ExecuteBorrow b(&parent,workspace,1024,stream);
        assert(!b.BeforeChild(&child,workspace+767,256,stream));assert(b.CancelBeforeParent());NoRetry(b);}
    Reset();{ExecuteBorrow b(&parent,workspace,1024,stream);Copy(b);
        assert(b.BeforeParent(&parent,workspace,768,stream));registered=nullptr;registered_bytes=0;
        assert(!b.AfterParent(17));assert(!b.CancelBeforeParent());NoRetry(b);assert(!resets);}
    std::cout<<"PASS_CPU_TND28_EXECUTE_BORROW_NO_DEVICE_NO_F\n";
}
