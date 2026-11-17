#ifndef DCAT_POSTWRITE_H
#define DCAT_POSTWRITE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Public ACL transfer ABI. No OpCache layout or externally supplied executor. */
uint32_t dcat_pw_abi(void);
/* The caller retains the actual tensor until restore/release succeeds. */
int dcat_pw_open(void *tensor, uint64_t bytes, int32_t device,
                 uint64_t owner, uint64_t generation, uint64_t *handle);
/* BF16 only. 0=lossless,1=+Inf,2=-Inf,3=qNaN,4=finite sign flip,5=zero. */
int dcat_pw_apply(uint64_t handle, uint64_t owner, uint64_t generation,
                  uint32_t mode, const uint64_t *indices, uint64_t count,
                  uint64_t *changed);
int dcat_pw_restore(uint64_t handle, uint64_t owner, uint64_t generation);
int dcat_pw_release(uint64_t handle, uint64_t owner, uint64_t generation);
/* 1 when the exact lease still exists. -1 means a different owner. */
int dcat_pw_live(uint64_t handle, uint64_t owner, uint64_t generation);
#ifdef __cplusplus
}
#endif
#endif
