#ifndef DCAT26_FIA_TYPED_DESCRIPTOR_V2_H
#define DCAT26_FIA_TYPED_DESCRIPTOR_V2_H
// Additive FIA-only wire records. Inventory v1 and QMM v2 remain independent.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <type_traits>

namespace dcat26_fia_typed {
constexpr uint32_t kVersion=2,kNcl=47,kBf16=27,kApiFiaV3=1;
enum Phase:uint32_t { OriginalBeforeGws=1,OriginalAndEffectiveAfterGws=2,OwnedEffectiveBeforeExecute=3 };
struct Descriptor {
    uint32_t version{kVersion},size{sizeof(Descriptor)},complete{},contiguous{};
    int32_t dtype{-1},format{-1};
    uint32_t view_rank{},storage_rank{},stride_rank{},reserved{};
    uint64_t descriptor{},raw{},view_bytes{},storage_bytes{},element_bytes{};
    int64_t offset{},view[8]{},storage[8]{},strides[8]{};
};
struct Record {
    uint32_t version{kVersion},size{sizeof(Record)},phase{},api{kApiFiaV3};
    int32_t device{-1},status{};
    uint64_t sequence{},event_sequence{},call{},thread{},executor{},registration{};
    uint32_t layout_length{},reserved{};
    char layout[16]{};
    Descriptor original{},effective{};
};
struct State {
    uint32_t version{kVersion},size{sizeof(State)};
    uint64_t capacity{},total{},stored{},overflow{},errors{},exceptions{};
    uint64_t before_gws{},after_gws{},before_execute{};
    uint64_t preparing{},prepared{},executing{};
};
inline bool same_descriptor(const Descriptor& a,const Descriptor& b,bool address=true,bool identity=true) {
    return a.version==kVersion && b.version==kVersion && a.size==sizeof(Descriptor) && b.size==sizeof(Descriptor)
        && a.complete==1 && b.complete==1 && a.contiguous==b.contiguous && !a.reserved && !b.reserved
        && a.dtype==b.dtype && a.format==b.format && a.view_rank==b.view_rank && a.storage_rank==b.storage_rank
        && a.stride_rank==b.stride_rank && a.view_bytes==b.view_bytes && a.storage_bytes==b.storage_bytes
        && a.element_bytes==b.element_bytes && a.offset==b.offset && (!address || a.raw==b.raw)
        && (!identity || a.descriptor==b.descriptor) && std::equal(a.view,a.view+8,b.view)
        && std::equal(a.storage,a.storage+8,b.storage) && std::equal(a.strides,a.strides+8,b.strides);
}
inline bool ncl(const Descriptor& value) {
    if(value.version!=kVersion || value.size!=sizeof(Descriptor) || value.complete!=1 || value.contiguous!=1
        || value.reserved || value.dtype!=static_cast<int32_t>(kBf16) || value.format!=static_cast<int32_t>(kNcl)
        || value.view_rank!=3 || value.storage_rank!=3 || value.stride_rank!=3 || value.offset || value.element_bytes!=2
        || !value.descriptor || !value.raw || value.raw>std::numeric_limits<uint64_t>::max()-value.storage_bytes)return false;
    uint64_t elements=1;
    for(unsigned i=3;i>0;--i) {
        const auto dimension=value.view[i-1];
        if(dimension<=0 || dimension!=value.storage[i-1] || value.strides[i-1]!=static_cast<int64_t>(elements)
            || static_cast<uint64_t>(dimension)>4*1024*1024/elements)return false;
        elements*=static_cast<uint64_t>(dimension);
    }
    for(unsigned i=3;i<8;++i)if(value.view[i] || value.storage[i] || value.strides[i])return false;
    return value.view_bytes==elements*2 && value.storage_bytes==value.view_bytes;
}
inline bool path(const Record& value) {
    return value.version==kVersion && value.size==sizeof(Record) && value.api==kApiFiaV3 && !value.reserved
        && value.device>=0 && value.event_sequence && value.call && value.thread
        && value.layout_length>0 && value.layout_length<sizeof(value.layout)
        && value.layout[value.layout_length]=='\0'
        && std::all_of(value.layout,value.layout+value.layout_length,[](char c){return c>='A' && c<='Z';})
        && std::all_of(value.layout+value.layout_length,value.layout+sizeof(value.layout),[](char c){return c=='\0';});
}
inline bool same_path(const Record& a,const Record& b) {
    return path(a) && path(b) && a.api==b.api && a.device==b.device && a.call==b.call && a.thread==b.thread
        && a.layout_length==b.layout_length && std::equal(a.layout,a.layout+sizeof(a.layout),b.layout);
}
inline bool same_record(const Record& a,const Record& b) {
    return same_path(a,b) && a.phase==b.phase && a.status==b.status && a.sequence==b.sequence
        && a.event_sequence==b.event_sequence && a.executor==b.executor && a.registration==b.registration
        && same_descriptor(a.original,b.original) && same_descriptor(a.effective,b.effective);
}
template<size_t Capacity> class Inventory {
    static_assert(std::is_trivially_copyable<Record>::value,"Typed descriptor wire record must be POD");
    std::mutex mutex_;
    State state_{};
    std::array<std::array<unsigned char,sizeof(Record)>,Capacity> records_{};
public:
    Inventory(){state_.capacity=Capacity;}
    void error(bool exception=false){std::lock_guard<std::mutex> lock(mutex_);++state_.errors;if(exception)++state_.exceptions;}
    uint64_t append(Record value) {
        std::lock_guard<std::mutex> lock(mutex_);value.sequence=++state_.total;
        if(value.phase==OriginalBeforeGws)++state_.before_gws;
        else if(value.phase==OriginalAndEffectiveAfterGws)++state_.after_gws;
        else if(value.phase==OwnedEffectiveBeforeExecute)++state_.before_execute;
        else ++state_.errors;
        if(state_.stored<Capacity)std::memcpy(records_[state_.stored++].data(),&value,sizeof(value));else ++state_.overflow;
        return value.sequence;
    }
    State state(){std::lock_guard<std::mutex> lock(mutex_);return state_;}
    bool read(uint64_t index,Record* out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!out || index>=state_.stored)return false;
        std::memcpy(out,records_[index].data(),sizeof(*out));return true;
    }
};
}
#endif
