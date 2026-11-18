#ifndef DCAT_POSTWRITE_H
#define DCAT_POSTWRITE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Public ACL transfer ABI. No OpCache layout or externally supplied executor. */
uint32_t dcat_pw_abi(void);
/* Actual linked function addresses only; no resolver, ACL call or lease. */
int dcat_pw_linked_address_v1(uint32_t slot, uint64_t *output);
/* Maximum disjoint spans for one owner, generation, thread and device. */
uint32_t dcat_pw_lease_capacity(void);
/* Total retained span bytes, counting each device span once. */
uint64_t dcat_pw_lease_byte_capacity(void);
/* Read-only bounded evidence copy; no lease, write or execution authority. */
int dcat_pw_read_span(const void *tensor, uint64_t bytes, int32_t device, void *host);
/* The caller retains the actual tensor until restore/release succeeds. */
int dcat_pw_open(void *tensor, uint64_t bytes, int32_t device,
                 uint64_t owner, uint64_t generation, uint64_t *handle);
/* BF16 only. 0=lossless,1=+Inf,2=-Inf,3=qNaN,4=finite sign flip,5=zero. */
int dcat_pw_apply(uint64_t handle, uint64_t owner, uint64_t generation,
                  uint32_t mode, const uint64_t *indices, uint64_t count,
                  uint64_t *changed);
int dcat_pw_restore(uint64_t handle, uint64_t owner, uint64_t generation);
int dcat_pw_release(uint64_t handle, uint64_t owner, uint64_t generation);
/* 1 for the exact live lease, 0 for an absent handle, -1 for a foreign scope. */
int dcat_pw_live(uint64_t handle, uint64_t owner, uint64_t generation);
#ifdef __cplusplus
}
#endif
#endif
