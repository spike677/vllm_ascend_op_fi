#ifndef DCAT_CPU_FAKE_ACL_H
#define DCAT_CPU_FAKE_ACL_H
#include <stddef.h>
#include <stdint.h>

enum { ACL_SUCCESS = 0, ACL_MEMCPY_DEVICE_TO_HOST = 1, ACL_MEMCPY_HOST_TO_DEVICE = 2 };
int aclrtGetDevice(int32_t *device);
int aclrtSynchronizeDevice();
int aclrtMemcpy(void *destination, size_t capacity, const void *source, size_t bytes, int direction);
#endif
