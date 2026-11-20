#ifndef DEMONCAT_TND28_EXECUTE_BORROW_H
#define DEMONCAT_TND28_EXECUTE_BORROW_H
// This local object must live inside the original generated Execute function.
// It grants no tiling/F permission. No handle is retained after Leave().
#ifndef DCAT_TND28_BORROW_CPU_FIXTURE
#include "opdev/op_executor.h"
#include "acl/acl_rt.h"
#include "acl_meta.h"
#endif
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sys/syscall.h>
#include <unistd.h>

extern "C" aclnnStatus NnopbaseGetUnContExecutor(void*, aclOpExecutor**, size_t*);
extern "C" aclnnStatus NnopbaseSetUnContExecutor(void*, aclOpExecutor*, size_t);
extern "C" aclnnStatus NnopbaseGetViewCopyExecutor(void*, aclOpExecutor**);

namespace tnd28_execute_borrow {
enum Error : uint32_t {
    NONE=0, OWNER=1, ORDER=2, SPAN=3, CHILD=4, VIEW_COPY=5,
    CHILD_CALL_FAILED=6, PARENT_CALL_FAILED=7, DRAIN=8,
    DETACH=9, RESET=10, RESET_READBACK=11, UNKNOWN_RETURN=12
};
struct Receipt {
    uint32_t version=1, size=sizeof(Receipt), first_error=0;
    bool child_started=false, child_consumed=false, parent_started=false, parent_returned=false;
    bool drained=false, detached=false, reset_attempted=false, reset_returned=false;
    bool cancelled=false, complete=false, terminal_uncertain=false, left=false;
    int32_t child_status=0, parent_status=0, drain_status=0, detach_status=0, reset_status=0;
    uint64_t pid=0, tid=0;
};

class ExecuteBorrow final {
public:
    // Only the actual generated Execute entry supplies these arguments. The
    // parent/child relation is re-read from NNOPBASE, not accepted from JSON.
    ExecuteBorrow(aclOpExecutor* parent, void* workspace, uint64_t bytes, aclrtStream stream)
        : parent_(parent), workspace_(workspace), bytes_(bytes), stream_(stream) {
        result_.pid=static_cast<uint64_t>(getpid());result_.tid=static_cast<uint64_t>(syscall(SYS_gettid));
        if (!parent_ || !workspace_ || !bytes_ || !stream_ ||
            Address(workspace_)>std::numeric_limits<uint64_t>::max()-bytes_) {Fail(SPAN);return;}
        aclOpExecutor* view=nullptr;
        if (NnopbaseGetUnContExecutor(parent_, &child_, &child_bytes_) || !child_ || !child_bytes_ || child_bytes_>bytes_)
            {Fail(CHILD);return;}
        if (NnopbaseGetViewCopyExecutor(parent_, &view) || view) {Fail(VIEW_COPY);return;}
        // Pinned CANN 8.5.1 ordinary, uncached child only. Never change its
        // repeatability to force a lifetime rule onto another executor type.
        if (child_->GetMagicNumber()!=K_EXECUTOR_MAGIC_NUMBER || child_->IsRepeatable()) {Fail(CHILD);return;}
        valid_=true;
    }
    ExecuteBorrow(const ExecuteBorrow&)=delete;
    ExecuteBorrow& operator=(const ExecuteBorrow&)=delete;
    ~ExecuteBorrow() {Leave();}

    bool BeforeChild(aclOpExecutor* child, void* workspace, uint64_t bytes, aclrtStream stream) {
        if (!Owner() || !valid_ || result_.first_error || result_.child_started || result_.parent_started)
            return Fail(ORDER);
        if (child!=child_ || bytes!=child_bytes_ || stream!=stream_ ||
            Address(workspace)!=Address(workspace_)+bytes_-child_bytes_) return Fail(SPAN);
        aclOpExecutor* current=nullptr;size_t current_bytes=0;
        if (NnopbaseGetUnContExecutor(parent_, &current, &current_bytes) || current!=child_ || current_bytes!=child_bytes_)
            return Fail(CHILD);
        result_.child_started=true; // Before invoking aclnnContiguous once.
        return true;
    }
    bool AfterChild(aclnnStatus status) {
        if (!Owner() || !result_.child_started || result_.child_consumed || result_.parent_started)
            return Fail(ORDER);
        // The actual libopapi_math -> CommonOpExecutorRun ordinary path has
        // consumed this child on every normal return, including a status error.
        // Do not inspect child_ or call its destructor from this point onward.
        result_.child_status=status;result_.child_consumed=true;child_=nullptr;
        if (status) Fail(CHILD_CALL_FAILED);
        return status==0;
    }
    // Drain prepared child data before the second-stage input check. This is
    // still cancellable: no permission to enqueue the parent is granted here.
    bool PrepareParent(aclOpExecutor* parent, void* workspace, uint64_t bytes, aclrtStream stream) {
        if (!Owner() || !result_.child_consumed || parent_prepared_ || result_.parent_started
            || result_.reset_attempted || result_.first_error || result_.complete)
            return Fail(ORDER);
        if (parent!=parent_ || workspace!=workspace_ || bytes!=bytes_-child_bytes_ || stream!=stream_)
            return Fail(SPAN);
        if (!Drain()) return false;
        parent_prepared_=true;
        return true;
    }
    bool CommitParent(aclOpExecutor* parent, void* workspace, uint64_t bytes, aclrtStream stream) {
        if (!Owner() || !parent_prepared_ || !result_.drained || result_.parent_started
            || result_.reset_attempted || result_.first_error || result_.complete)
            return Fail(ORDER);
        if (parent!=parent_ || workspace!=workspace_ || bytes!=bytes_-child_bytes_ || stream!=stream_)
            return Fail(SPAN);
        result_.parent_started=true;result_.drained=false;
        return true;
    }
    bool BeforeParent(aclOpExecutor* parent, void* workspace, uint64_t bytes, aclrtStream stream) {
        return PrepareParent(parent,workspace,bytes,stream) && CommitParent(parent,workspace,bytes,stream);
    }
    bool AfterParent(aclnnStatus status) {
        if (!Owner() || !result_.parent_started || result_.parent_returned) return Fail(ORDER);
        result_.parent_status=status;result_.parent_returned=true;
        if (status) Fail(PARENT_CALL_FAILED);
        // Do not destroy this parent: the original normal-return path owns its
        // reset. On a status failure only drain is attempted; no guessed reuse.
        if (!Drain()) return false;
        if (status) {result_.terminal_uncertain=true;return false;}
        if (!ZeroChildren()) {Fail(RESET_READBACK);result_.terminal_uncertain=true;return false;}
        result_.complete=true;
        return true;
    }
    bool CancelBeforeParent() {
        if (!Owner() || !valid_ || result_.parent_started || result_.left || result_.reset_attempted || result_.terminal_uncertain)
            return Fail(ORDER);
        if (result_.child_started && !result_.child_consumed) {
            Fail(UNKNOWN_RETURN);result_.terminal_uncertain=true;return false;
        }
        if (!Drain()) return false;
        if (result_.child_consumed && !result_.detached) {
            // The parent's stored child address is now dangling. It may only
            // be cleared; it must not be passed to parent Reset as a live child.
            result_.detach_status=NnopbaseSetUnContExecutor(parent_, nullptr, 0);
            const bool zero=ZeroChildren();
            if (zero) result_.detached=true; // Record completion before any receipt I/O.
            if (result_.detach_status || !zero) Fail(DETACH);
            if (!zero) {result_.terminal_uncertain=true;return false;}
        }
        // Before-copy cancellation leaves the still-live child owned by the
        // parent. Public destruction consumes it exactly once on that path.
        result_.reset_attempted=true;
        result_.reset_status=aclDestroyAclOpExecutor(parent_);
        if (result_.reset_status) {Fail(RESET);result_.terminal_uncertain=true;return false;}
        result_.reset_returned=true;child_=nullptr; // Never destroy again, even if readback fails.
        if (!ZeroChildren()) {Fail(RESET_READBACK);result_.terminal_uncertain=true;return false;}
        result_.cancelled=true;result_.complete=true;
        return true;
    }
    void OriginalThrew() noexcept {Fail(UNKNOWN_RETURN);result_.terminal_uncertain=true;}
    void Leave() noexcept {
        if (result_.left) return;
        // No SDK operation is legal from a later RPC/destructor retry. An
        // incomplete call retains an uncertain receipt, not operational handles.
        if (!result_.complete) {Fail(UNKNOWN_RETURN);result_.terminal_uncertain=true;}
        parent_=child_=nullptr;workspace_=nullptr;stream_=nullptr;bytes_=child_bytes_=0;
        result_.left=true;
    }
    Receipt Snapshot() const noexcept {return result_;}
private:
    static uint64_t Address(const void* p) noexcept {return reinterpret_cast<uint64_t>(p);}
    bool Owner() noexcept {
        return !result_.left && result_.pid==static_cast<uint64_t>(getpid()) &&
            result_.tid==static_cast<uint64_t>(syscall(SYS_gettid)) ? true : Fail(OWNER);
    }
    bool Fail(Error error) noexcept {if (!result_.first_error)result_.first_error=error;return false;}
    bool ZeroChildren() {
        aclOpExecutor* child=nullptr;aclOpExecutor* view=nullptr;size_t bytes=0;
        return NnopbaseGetUnContExecutor(parent_, &child, &bytes)==0 && !child && !bytes &&
            NnopbaseGetViewCopyExecutor(parent_, &view)==0 && !view;
    }
    bool Drain() {
        if (result_.drained)return true;
        result_.drain_status=aclrtSynchronizeStream(stream_);
        if (result_.drain_status) {Fail(DRAIN);result_.terminal_uncertain=true;return false;}
        result_.drained=true;return true;
    }
    aclOpExecutor* parent_=nullptr;
    aclOpExecutor* child_=nullptr;
    void* workspace_=nullptr;
    uint64_t bytes_=0;
    size_t child_bytes_=0;
    aclrtStream stream_=nullptr;
    bool valid_=false, parent_prepared_=false;
    Receipt result_{};
};
} // namespace tnd28_execute_borrow
#endif
