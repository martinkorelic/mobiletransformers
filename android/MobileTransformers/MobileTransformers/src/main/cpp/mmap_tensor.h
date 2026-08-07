//
// #12: RAII memory-mapped file region for zero-copy external-initializer loading (Gate 0.2 experiment).
// Owned by the weight cache and unmapped in its destructor — NOT eagerly freed (the mapped bytes must
// outlive any Ort::Value that points into them). Default-off; the copy path stays the shipping default.
//

#ifndef MOBILETRANSFORMERS_MMAP_TENSOR_H
#define MOBILETRANSFORMERS_MMAP_TENSOR_H

#include <cstddef>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "logging.h"

// A single mmap'd file. Move-only; unmaps on destruction.
class MmapRegion {
public:
    MmapRegion() = default;

    // Map the whole file read-only/private. On failure, data() == nullptr and size() == 0.
    explicit MmapRegion(const std::string& path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            LOGE("mmap: open failed for %s", path.c_str());
            return;
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
            ::close(fd);
            LOGE("mmap: fstat failed / empty for %s", path.c_str());
            return;
        }
        size_ = static_cast<size_t>(st.st_size);
        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);  // the mapping keeps its own reference; fd can close.
        if (p == MAP_FAILED) {
            data_ = nullptr;
            size_ = 0;
            LOGE("mmap: mmap failed for %s", path.c_str());
            return;
        }
        data_ = p;
    }

    ~MmapRegion() { reset(); }

    MmapRegion(MmapRegion&& other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    MmapRegion& operator=(MmapRegion&& other) noexcept {
        if (this != &other) {
            reset();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    MmapRegion(const MmapRegion&) = delete;
    MmapRegion& operator=(const MmapRegion&) = delete;

    bool valid() const { return data_ != nullptr && size_ > 0; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }

private:
    void reset() {
        if (data_ != nullptr) {
            ::munmap(data_, size_);
            data_ = nullptr;
            size_ = 0;
        }
    }

    void* data_ = nullptr;
    size_t size_ = 0;
};

#endif  // MOBILETRANSFORMERS_MMAP_TENSOR_H
