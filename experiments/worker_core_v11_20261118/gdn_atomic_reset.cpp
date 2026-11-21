// Diagnostic state reset: no tensor arguments, reads or writes.
// Ascend910B4 only (20 AIC / 40 AIV); never grants fault admission.
#include "kernel_operator.h"
#include <cstdint>
extern "C" __global__ __cube__ void dcat_gdn_reset_cube() {
    set_atomic_none();
    pipe_barrier(PIPE_ALL);
}
extern "C" __global__ __vector__ void dcat_gdn_reset_vector() {
    set_atomic_none();
    pipe_barrier(PIPE_ALL);
}
extern "C" void dcat_gdn_reset_atomic_v1(void* stream) {
    dcat_gdn_reset_cube<<<20, nullptr, stream>>>();
    dcat_gdn_reset_vector<<<40, nullptr, stream>>>();
}
