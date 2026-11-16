// Included only in the opt-in deployment build. Actual FIA destination boundary.
#undef fault26_fia_writer_prepare_output
#include <thread>
struct Dcat26FiaCall {
    uint32_t version,size;uint64_t token;
    int32_t device,dtype;
    uint64_t output,output_bytes,residual,residual_bytes,producer_sequence;
    uint32_t metadata_ok,dimensions;int64_t shape[4];
};
using Dcat26FiaCallback=int (*)(const Dcat26FiaCall *);
static Dcat26FiaCallback dcat26f_callback=nullptr;
static std::thread::id dcat26f_owner;
static uint64_t dcat26f_next=0;
static thread_local uint64_t dcat26f_token=0;
static thread_local bool dcat26f_called=false;
static thread_local const aclTensor *dcat26f_original=nullptr,*dcat26f_result=nullptr;
extern "C" int dcat26f_install_v1(Dcat26FiaCallback callback,uint64_t session) {
    pthread_once(&g_once,initialize);
    std::lock_guard<std::mutex> lifecycle(g_lifecycle_mutex);
    std::lock_guard<std::mutex> control(g_control_mutex);
    if(!callback || !session || !g_runtime_ready || !g_allowed || dcat26f_callback || !g_pending_stomps.empty())return 1;
    // Arm before the next real producer; no synthetic warm-up FIA is used.
    g_session=session;g_matching_count=0;g_hit_count=0;g_hit_claimed=false;
    g_min_residual_sequence=g_add_rms_sequence.load(std::memory_order_relaxed);
    dcat26f_callback=callback;dcat26f_owner=std::this_thread::get_id();return 0;
}
extern "C" uint64_t dcat26f_pending_v1() {
    std::lock_guard<std::mutex> lock(g_lifecycle_mutex);return g_pending_stomps.size();
}
extern "C" int dcat26f_remove_v1(Dcat26FiaCallback callback) {
    std::lock_guard<std::mutex> lock(g_lifecycle_mutex);
    if(callback!=dcat26f_callback || !g_pending_stomps.empty() || g_prepared.active)return 1;
    dcat26f_callback=nullptr;return 0;
}
extern "C" int dcat26f_continue_v1(uint64_t token) {
    if(!token || token!=dcat26f_token || dcat26f_called)return -1;
    dcat26f_called=true;dcat26f_result=dcat26_fia_prepare_impl(dcat26f_original);
    return dcat26f_result!=dcat26f_original?1:0;
}
extern "C" const aclTensor *fault26_fia_writer_prepare_output(const aclTensor *output) {
    if(!dcat26f_callback || dcat26f_token)return dcat26_fia_prepare_impl(output);
    if(std::this_thread::get_id()!=dcat26f_owner)return output;
    Dcat26FiaCall packet{};packet.version=1;packet.size=sizeof(packet);packet.token=++dcat26f_next;
    packet.device=-1;g_get_device(&packet.device);
    auto metadata=capture_tensor(output);
    packet.metadata_ok=metadata.ok;packet.dtype=metadata.dtype;
    packet.output=reinterpret_cast<uint64_t>(metadata.address);packet.output_bytes=metadata.bytes;
    packet.dimensions=metadata.view_dims.size();
    for(size_t i=0;i<std::min<size_t>(4,metadata.view_dims.size());++i)packet.shape[i]=metadata.view_dims[i];
    {
        std::lock_guard<std::mutex> lock(g_lifecycle_mutex);
        auto found=g_current_residuals.find(packet.device);
        if(found!=g_current_residuals.end() && found->second.tensor.ok) {
            packet.residual=reinterpret_cast<uint64_t>(found->second.tensor.address);
            packet.residual_bytes=found->second.tensor.bytes;packet.producer_sequence=found->second.producer_sequence;
        }
    }
    dcat26f_token=packet.token;dcat26f_called=false;dcat26f_original=output;dcat26f_result=output;
    dcat26f_callback(&packet);
    const aclTensor *result=dcat26f_result;
    dcat26f_token=0;dcat26f_original=nullptr;dcat26f_result=nullptr;
    return result;
}
