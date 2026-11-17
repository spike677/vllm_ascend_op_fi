// HOST TEST ONLY. Root's separate build must use the actual CANN SDK header.
#ifndef DCAT_QUANT_TEST_DYNAMIC_QUANT_H
#define DCAT_QUANT_TEST_DYNAMIC_QUANT_H
#include <acl/acl.h>
extern "C" aclnnStatus aclnnDynamicQuantV2GetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,int64_t,
    const aclTensor*,const aclTensor*,const aclTensor*,uint64_t*,aclOpExecutor**);
extern "C" aclnnStatus aclnnDynamicQuantV2(void*,uint64_t,aclOpExecutor*,aclrtStream);
#endif
