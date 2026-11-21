// Diagnostic helper linked into original C220 GDN kernels, no tensor access.
#include "kernel_operator.h"
extern "C" __attribute__((always_inline)) __attribute__((cce_aicore))
void dcat_gdn_atomic_none_aic() { set_atomic_none(); }
extern "C" __attribute__((always_inline)) __attribute__((cce_aicore))
void dcat_gdn_atomic_none_aiv() { set_atomic_none(); }
