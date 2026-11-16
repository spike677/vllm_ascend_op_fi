// Included only in the opt-in deployment build. Original arithmetic unchanged.
#undef fault26_qmm_underflow_prepare
#include <thread>
using Dcat26QmmCallback = int (*)(const Fault26QmmUnderflowContext *, uint64_t);
static Dcat26QmmCallback dcat26q_callback=nullptr;
static std::thread::id dcat26q_owner;
static uint64_t dcat26q_next=0;
static thread_local uint64_t dcat26q_token=0;
static thread_local bool dcat26q_called=false, dcat26q_result=false;
static thread_local const Fault26QmmUnderflowContext *dcat26q_context=nullptr;
static thread_local aclrtStream dcat26q_stream=nullptr;
static thread_local Fault26QmmUnderflowDecision *dcat26q_decision=nullptr;
extern "C" int dcat26q_install_v1(Dcat26QmmCallback callback) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if(!callback || dcat26q_callback || g_pending.active)return 1;
    dcat26q_callback=callback;dcat26q_owner=std::this_thread::get_id();return 0;
}
extern "C" int dcat26q_remove_v1(Dcat26QmmCallback callback) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if(callback!=dcat26q_callback || g_pending.active)return 1;
    dcat26q_callback=nullptr;return 0;
}
extern "C" int dcat26q_continue_v1(uint64_t token) {
    if(!token || token!=dcat26q_token || dcat26q_called)return -1;
    dcat26q_called=true;
    dcat26q_result=dcat26_qmm_prepare_impl(dcat26q_context,dcat26q_stream,dcat26q_decision);
    return dcat26q_result?1:0;
}
extern "C" uint64_t dcat26q_pending_v1() {
    std::lock_guard<std::mutex> lock(g_mutex);return g_pending.active?g_pending.token:0;
}
extern "C" bool fault26_qmm_underflow_prepare(const Fault26QmmUnderflowContext *context,
    aclrtStream stream,Fault26QmmUnderflowDecision *decision) {
    if(!dcat26q_callback || dcat26q_token)return dcat26_qmm_prepare_impl(context,stream,decision);
    if(std::this_thread::get_id()!=dcat26q_owner){if(decision)*decision=Fault26QmmUnderflowDecision{};return false;}
    dcat26q_context=context;dcat26q_stream=stream;dcat26q_decision=decision;
    dcat26q_called=false;dcat26q_result=false;dcat26q_token=++dcat26q_next;
    int status=dcat26q_callback(context,dcat26q_token);
    bool result=dcat26q_called && dcat26q_result;
    // Do not retry original on callback failure. The model owner records the
    // error and quarantines the process if any live recovery remains pending.
    if(status && !dcat26q_called && decision)*decision=Fault26QmmUnderflowDecision{};
    dcat26q_context=nullptr;dcat26q_decision=nullptr;dcat26q_stream=nullptr;dcat26q_token=0;
    return result;
}
