// src/platform/direct_file.cpp - see include/strata/platform/direct_file.hpp.
#include "strata/platform/direct_file.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__linux__) && __has_include(<linux/io_uring.h>)
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#if defined(__NR_io_uring_setup) && defined(__NR_io_uring_enter) && defined(IORING_FEAT_RW_CUR_POS)
#define STRATA_HAVE_IO_URING 1
#endif
#endif
#endif
#ifndef STRATA_HAVE_IO_URING
#define STRATA_HAVE_IO_URING 0
#endif

namespace strata::platform {

double now_us() {
    using namespace std::chrono;
    return (double) duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

void* DirectFile::alloc_aligned(size_t bytes) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* p = nullptr;
    return posix_memalign(&p, alignment(), bytes) == 0 ? p : nullptr;
#endif
}

void DirectFile::free_aligned(void* p) {
    if (p == nullptr) return;
#if defined(_WIN32)
    VirtualFree(p, 0, MEM_RELEASE);
#else
    std::free(p);
#endif
}

#if defined(_WIN32)
// ------------------------------------------------------------------------------------------------ Windows
namespace {
/// One in-flight request. The OVERLAPPED must be the first member so a completion packet's OVERLAPPED*
/// converts back to the request.
struct Req {
    OVERLAPPED ov;
    uint64_t tag;
};
}  // namespace

struct DirectFile::Impl {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE port = nullptr;
    uint64_t size = 0;
    std::deque<Req*> free_reqs;
    std::vector<Req*> all_reqs;
    std::deque<Completion> immediate;   // requests that failed after being counted as queued

    Req* take() {
        if (free_reqs.empty()) {
            Req* r = new Req();
            all_reqs.push_back(r);
            return r;
        }
        Req* r = free_reqs.front();
        free_reqs.pop_front();
        return r;
    }
};

DirectFile::DirectFile() : impl_(new Impl) {}
DirectFile::~DirectFile() {
    close();
    for (Req* r : impl_->all_reqs) delete r;
    delete impl_;
}

bool DirectFile::open(const std::string& path, std::string& err, uint32_t) {
    close();
    const int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath((size_t) (wlen > 0 ? wlen : 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);
    // NO_BUFFERING: the cache manager never holds these pages, so the table cannot grow into RAM.
    // RANDOM_ACCESS: no read-ahead. OVERLAPPED: many reads in flight, completed through the port below.
    impl_->file = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (impl_->file == INVALID_HANDLE_VALUE) {
        err = "DirectFile: cannot open " + path + " (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(impl_->file, &sz)) {
        err = "DirectFile: cannot size " + path;
        close();
        return false;
    }
    impl_->size = (uint64_t) sz.QuadPart;
    impl_->port = CreateIoCompletionPort(impl_->file, nullptr, 0, 1);
    if (impl_->port == nullptr) {
        err = "DirectFile: CreateIoCompletionPort failed (error " + std::to_string(GetLastError()) + ")";
        close();
        return false;
    }
    // Completions of reads that finish synchronously must still be queued to the port, so every submit
    // produces exactly one packet and `wait` is the only completion path.
    return true;
}

void DirectFile::close() {
    if (impl_->port != nullptr) CloseHandle(impl_->port);
    if (impl_->file != INVALID_HANDLE_VALUE) CloseHandle(impl_->file);
    impl_->port = nullptr;
    impl_->file = INVALID_HANDLE_VALUE;
    impl_->size = 0;
    impl_->immediate.clear();
}

bool DirectFile::is_open() const { return impl_->file != INVALID_HANDLE_VALUE; }
uint64_t DirectFile::size() const { return impl_->size; }
const char* DirectFile::backend() const { return "iocp"; }

bool DirectFile::submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err) {
    if (!is_open()) { err = "DirectFile: not open"; return false; }
    if (offset % alignment() || length % alignment() || ((uintptr_t) buffer) % alignment() || length == 0) {
        err = "DirectFile: unaligned request";
        return false;
    }
    Req* r = impl_->take();
    std::memset(&r->ov, 0, sizeof r->ov);
    r->ov.Offset = (DWORD) (offset & 0xFFFFFFFFull);
    r->ov.OffsetHigh = (DWORD) (offset >> 32);
    r->tag = tag;
    if (!ReadFile(impl_->file, buffer, length, nullptr, &r->ov)) {
        const DWORD e = GetLastError();
        if (e != ERROR_IO_PENDING) {
            impl_->free_reqs.push_back(r);
            if (e == ERROR_HANDLE_EOF) {           // at or past end of file: a zero-byte completion
                impl_->immediate.push_back(Completion{tag, 0, true});
                return true;
            }
            err = "DirectFile: ReadFile failed (error " + std::to_string(e) + ")";
            return false;
        }
    }
    return true;
}

int DirectFile::wait(Completion* out, int max, int timeout_ms) {
    int n = 0;
    while (n < max && !impl_->immediate.empty()) {
        out[n++] = impl_->immediate.front();
        impl_->immediate.pop_front();
    }
    if (n == max || !is_open()) return n;
    OVERLAPPED_ENTRY entries[64];
    const ULONG want = (ULONG) (max - n < 64 ? max - n : 64);
    ULONG got = 0;
    const DWORD t = timeout_ms < 0 ? INFINITE : (DWORD) timeout_ms;
    if (!GetQueuedCompletionStatusEx(impl_->port, entries, want, &got, n > 0 ? 0 : t, FALSE)) return n;
    for (ULONG i = 0; i < got; ++i) {
        if (entries[i].lpOverlapped == nullptr) {          // a wake() packet, not a read
            out[n++] = Completion{WAKE_TAG, 0, true};
            continue;
        }
        Req* r = (Req*) entries[i].lpOverlapped;
        const uint32_t status = (uint32_t) r->ov.Internal;   // an NTSTATUS
        Completion c;
        c.tag = r->tag;
        c.bytes = entries[i].dwNumberOfBytesTransferred;
        // STATUS_END_OF_FILE (0xC0000011) is a legal short read at the table's last page.
        c.ok = status == 0 || status == 0xC0000011u;
        out[n++] = c;
        impl_->free_reqs.push_back(r);
    }
    return n;
}

void DirectFile::wake() {
    if (impl_->port != nullptr) PostQueuedCompletionStatus(impl_->port, 0, 0, nullptr);
}

#else
// ------------------------------------------------------------------------------------------------ POSIX
// Issue #15: a read no longer completes inside `submit` (which made the reads of a window serial, `wake` a no-op
// and `--ple-inflight` meaningless). Three backends, one contract - a queued request produces exactly one
// completion, and `wake` makes a blocked `wait` return:
//   uring    io_uring through raw syscalls (no liburing), the default where the kernel allows it;
//   threads  a pool of pread threads, min(queue depth, 16), when io_uring is unavailable;
//   sync     pread inside `submit`, the previous path, kept as the A/B arm.
// STRATA_PLE_IO_BACKEND=uring|threads|sync picks one. `submit` is called from one thread at a time and `wait` from
// one thread at a time (PleReader: its worker, or the caller in caller-thread mode); `wake` from any thread.
namespace {

struct PosixReq {
    uint64_t offset;
    void* buffer;
    uint32_t length;
    uint64_t tag;
};

Completion pread_one(int fd, const PosixReq& r) {
    ssize_t got;
    do {
        got = pread(fd, r.buffer, r.length, (off_t) r.offset);
    } while (got < 0 && errno == EINTR);
    return Completion{r.tag, got < 0 ? 0u : (uint32_t) got, got >= 0};
}

enum class Backend { Sync, Threads, Uring };

#if STRATA_HAVE_IO_URING
/// The two rings and the SQE array, mapped from the ring fd. Only the fields the reads need.
struct Uring {
    int fd = -1;
    uint32_t sq_entries = 0, cq_entries = 0;
    uint8_t* sq_ring = nullptr;
    size_t sq_ring_bytes = 0;
    uint8_t* cq_ring = nullptr;            // == sq_ring with IORING_FEAT_SINGLE_MMAP
    size_t cq_ring_bytes = 0;
    io_uring_sqe* sqes = nullptr;
    size_t sqes_bytes = 0;
    uint32_t *sq_head = nullptr, *sq_tail = nullptr, *sq_mask = nullptr, *sq_array = nullptr;
    uint32_t *cq_head = nullptr, *cq_tail = nullptr, *cq_mask = nullptr;
    io_uring_cqe* cqes = nullptr;

    int enter(unsigned to_submit, unsigned min_complete, unsigned flags) {
        return (int) syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags, nullptr, (size_t) 0);
    }

    bool setup(uint32_t entries) {
        io_uring_params p;
        std::memset(&p, 0, sizeof p);
        fd = (int) syscall(__NR_io_uring_setup, entries, &p);
        if (fd < 0) return false;
        // IORING_OP_READ arrived in 5.6 together with this feature bit; an older kernel takes the thread pool.
        if (!(p.features & IORING_FEAT_RW_CUR_POS)) { teardown(); return false; }
        sq_entries = p.sq_entries;
        cq_entries = p.cq_entries;
        sq_ring_bytes = p.sq_off.array + p.sq_entries * sizeof(uint32_t);
        cq_ring_bytes = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
        const bool single = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
        if (single) sq_ring_bytes = cq_ring_bytes = std::max(sq_ring_bytes, cq_ring_bytes);
        void* sq = mmap(nullptr, sq_ring_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
        if (sq == MAP_FAILED) { teardown(); return false; }
        sq_ring = (uint8_t*) sq;
        if (single) {
            cq_ring = sq_ring;
        } else {
            void* cq = mmap(nullptr, cq_ring_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd,
                            IORING_OFF_CQ_RING);
            if (cq == MAP_FAILED) { teardown(); return false; }
            cq_ring = (uint8_t*) cq;
        }
        sqes_bytes = p.sq_entries * sizeof(io_uring_sqe);
        void* se = mmap(nullptr, sqes_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
        if (se == MAP_FAILED) { sqes = nullptr; teardown(); return false; }
        sqes = (io_uring_sqe*) se;
        sq_head = (uint32_t*) (sq_ring + p.sq_off.head);
        sq_tail = (uint32_t*) (sq_ring + p.sq_off.tail);
        sq_mask = (uint32_t*) (sq_ring + p.sq_off.ring_mask);
        sq_array = (uint32_t*) (sq_ring + p.sq_off.array);
        cq_head = (uint32_t*) (cq_ring + p.cq_off.head);
        cq_tail = (uint32_t*) (cq_ring + p.cq_off.tail);
        cq_mask = (uint32_t*) (cq_ring + p.cq_off.ring_mask);
        cqes = (io_uring_cqe*) (cq_ring + p.cq_off.cqes);
        return true;
    }

    void teardown() {
        if (sqes != nullptr) munmap(sqes, sqes_bytes);
        if (cq_ring != nullptr && cq_ring != sq_ring) munmap(cq_ring, cq_ring_bytes);
        if (sq_ring != nullptr) munmap(sq_ring, sq_ring_bytes);
        if (fd >= 0) ::close(fd);
        *this = Uring{};
    }

    /// One SQE, handed to the kernel at once. The caller serializes pushes. False: the kernel did not take it,
    /// and it has been withdrawn from the ring (no completion will come for it).
    bool push(uint8_t opcode, int file, uint64_t off, void* buf, uint32_t len, uint64_t user) {
        const uint32_t tail = *sq_tail;                        // only this side writes the SQ tail
        if (tail - __atomic_load_n(sq_head, __ATOMIC_ACQUIRE) >= sq_entries) return false;
        const uint32_t i = tail & *sq_mask;
        io_uring_sqe* e = &sqes[i];
        std::memset(e, 0, sizeof *e);
        e->opcode = opcode;
        e->fd = file;
        e->off = off;
        e->addr = (uint64_t) (uintptr_t) buf;
        e->len = len;
        e->user_data = user;
        sq_array[i] = i;
        __atomic_store_n(sq_tail, tail + 1, __ATOMIC_RELEASE);
        for (int attempt = 0; attempt < 64; ++attempt) {
            const int r = enter(1, 0, 0);
            if (r >= 1 || __atomic_load_n(sq_head, __ATOMIC_ACQUIRE) != tail) return true;
            if (r < 0 && errno != EINTR && errno != EAGAIN && errno != EBUSY) break;
            if (r < 0 && errno != EINTR) sched_yield();
        }
        // Not consumed, and no other enter can consume it (to_submit is 0 everywhere else): withdraw it.
        __atomic_store_n(sq_tail, tail, __ATOMIC_RELEASE);
        return false;
    }
};
#endif

}  // namespace

struct DirectFile::Impl {
    int fd = -1;
    uint64_t size = 0;
    Backend backend = Backend::Sync;

    // Completed inside `submit`: every read in `sync`, and a read the ring could not take in `uring`.
    std::mutex done_mu;
    std::deque<Completion> done;

    // threads
    std::mutex mu;
    std::condition_variable cv_todo, cv_done;
    std::deque<PosixReq> todo;
    std::deque<Completion> pool_done;
    std::vector<std::thread> pool;
    uint64_t pool_pending = 0;             // queued or being read, not yet in `pool_done`
    bool pool_wake = false;
    bool pool_stop = false;

#if STRATA_HAVE_IO_URING
    Uring ring;
    std::mutex sq_mu;                      // serializes SQ pushes (`submit` and `wake`)
    std::atomic<uint32_t> ring_reads{0};   // reads in the kernel
    std::atomic<bool> wake_armed{false};   // one NOP in flight is enough to wake the waiter
    uint32_t ring_cap = 0;                 // reads in flight; the CQ holds twice the SQ, so never overflows
#endif

    void pool_loop() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv_todo.wait(lk, [&] { return pool_stop || !todo.empty(); });
            if (pool_stop) break;
            const PosixReq r = todo.front();
            todo.pop_front();
            lk.unlock();
            const Completion c = pread_one(fd, r);
            lk.lock();
            pool_done.push_back(c);
            --pool_pending;
            cv_done.notify_all();
        }
    }

    int take_done(Completion* out, int max) {
        std::lock_guard<std::mutex> lk(done_mu);
        int n = 0;
        while (n < max && !done.empty()) {
            out[n++] = done.front();
            done.pop_front();
        }
        return n;
    }

#if STRATA_HAVE_IO_URING
    int reap(Completion* out, int max) {
        uint32_t head = *ring.cq_head;                         // only the waiter writes the CQ head
        const uint32_t tail = __atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE);
        int n = 0;
        while (head != tail && n < max) {
            const io_uring_cqe& c = ring.cqes[head & *ring.cq_mask];
            if (c.user_data == WAKE_TAG) {
                wake_armed.store(false);
                out[n++] = Completion{WAKE_TAG, 0, true};
            } else {
                out[n++] = Completion{c.user_data, c.res < 0 ? 0u : (uint32_t) c.res, c.res >= 0};
                ring_reads.fetch_sub(1);
            }
            ++head;
        }
        __atomic_store_n(ring.cq_head, head, __ATOMIC_RELEASE);
        return n;
    }
#endif
};

DirectFile::DirectFile() : impl_(new Impl) {}
DirectFile::~DirectFile() { close(); delete impl_; }

bool DirectFile::open(const std::string& path, std::string& err, uint32_t queue_depth) {
    close();
    Impl& m = *impl_;
    m.fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (m.fd < 0) { err = "DirectFile: cannot open " + path; return false; }
    struct stat st;
    if (fstat(m.fd, &st) != 0) { err = "DirectFile: cannot size " + path; close(); return false; }
    m.size = (uint64_t) st.st_size;
    if (queue_depth == 0) queue_depth = 1;
    const char* env = std::getenv("STRATA_PLE_IO_BACKEND");
    const std::string want = env != nullptr ? env : "";
    if (want == "sync") {
        m.backend = Backend::Sync;
        return true;
    }
#if STRATA_HAVE_IO_URING
    if (want != "threads") {
        // SQ entries: the queue depth plus the wake NOP, a power of two; the kernel doubles it for the CQ.
        uint32_t entries = 8;
        while (entries < queue_depth + 2 && entries < 4096) entries <<= 1;
        if (m.ring.setup(entries) || (entries > 64 && m.ring.setup(64))) {
            m.ring_cap = m.ring.sq_entries - 1;
            m.ring_reads.store(0);
            m.wake_armed.store(false);
            m.backend = Backend::Uring;
            return true;
        }
        if (want == "uring")
            std::fprintf(stderr, "strata: io_uring is not available (errno %d); PLE reads use a pread pool\n", errno);
    }
#endif
    uint32_t threads = std::min<uint32_t>(queue_depth, 16);
    if (const char* t = std::getenv("STRATA_PLE_IO_THREADS")) {
        const int v = std::atoi(t);
        if (v >= 1 && v <= 64) threads = (uint32_t) v;
    }
    m.backend = Backend::Threads;
    m.pool_stop = false;
    m.pool_wake = false;
    m.pool_pending = 0;
    for (uint32_t i = 0; i < threads; ++i) m.pool.emplace_back([&m] { m.pool_loop(); });
    return true;
}

void DirectFile::close() {
    Impl& m = *impl_;
    if (!m.pool.empty()) {
        {
            std::lock_guard<std::mutex> lk(m.mu);
            m.pool_stop = true;                    // queued reads are dropped; a read in progress finishes first
        }
        m.cv_todo.notify_all();
        for (std::thread& t : m.pool) t.join();
        m.pool.clear();
    }
    m.todo.clear();
    m.pool_done.clear();
    m.pool_pending = 0;
    m.pool_wake = false;
    m.pool_stop = false;
#if STRATA_HAVE_IO_URING
    if (m.ring.fd >= 0) {
        // No read may still target a caller's buffer once close returns.
        while (m.ring_reads.load() > 0) {
            Completion c[64];
            if (m.reap(c, 64) == 0 && m.ring.enter(0, 1, IORING_ENTER_GETEVENTS) < 0 && errno != EINTR) break;
        }
        m.ring.teardown();
    }
    m.ring_reads.store(0);
    m.wake_armed.store(false);
    m.ring_cap = 0;
#endif
    if (m.fd >= 0) ::close(m.fd);
    m.fd = -1;
    m.size = 0;
    m.done.clear();
    m.backend = Backend::Sync;
}

bool DirectFile::is_open() const { return impl_->fd >= 0; }
uint64_t DirectFile::size() const { return impl_->size; }

const char* DirectFile::backend() const {
    switch (impl_->backend) {
    case Backend::Uring: return "uring";
    case Backend::Threads: return "threads";
    default: return "sync";
    }
}

bool DirectFile::submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err) {
    if (offset % alignment() || length % alignment() || ((uintptr_t) buffer) % alignment() || length == 0) {
        err = "DirectFile: unaligned request";
        return false;
    }
    Impl& m = *impl_;
    const PosixReq r{offset, buffer, length, tag};
    if (m.backend == Backend::Threads) {
        {
            std::lock_guard<std::mutex> lk(m.mu);
            m.todo.push_back(r);
            ++m.pool_pending;
        }
        m.cv_todo.notify_one();
        return true;
    }
#if STRATA_HAVE_IO_URING
    if (m.backend == Backend::Uring) {
        {
            std::lock_guard<std::mutex> lk(m.sq_mu);
            if (m.ring_reads.load() < m.ring_cap) {
                m.ring_reads.fetch_add(1);
                if (m.ring.push(IORING_OP_READ, m.fd, offset, buffer, length, tag)) return true;
                m.ring_reads.fetch_sub(1);
            }
        }
        // The ring is full or refused the read: complete it here, like the sync path.
        const Completion c = pread_one(m.fd, r);
        {
            std::lock_guard<std::mutex> lk(m.done_mu);
            m.done.push_back(c);
        }
        wake();                                    // a waiter blocked in the ring must look at `done`
        return true;
    }
#endif
    const Completion c = pread_one(m.fd, r);
    std::lock_guard<std::mutex> lk(m.done_mu);
    m.done.push_back(c);
    return true;
}

void DirectFile::wake() {
    Impl& m = *impl_;
    if (m.backend == Backend::Threads) {
        {
            std::lock_guard<std::mutex> lk(m.mu);
            m.pool_wake = true;
        }
        m.cv_done.notify_all();
        return;
    }
#if STRATA_HAVE_IO_URING
    if (m.backend == Backend::Uring) {
        if (m.wake_armed.exchange(true)) return;
        std::lock_guard<std::mutex> lk(m.sq_mu);
        // A refused NOP loses nothing but latency: the waiter blocks only while reads are in flight, and each
        // completion wakes it.
        if (!m.ring.push(IORING_OP_NOP, -1, 0, nullptr, 0, WAKE_TAG)) m.wake_armed.store(false);
    }
#endif
    // sync: reads complete inside submit; nothing ever blocks in wait
}

int DirectFile::wait(Completion* out, int max, int timeout_ms) {
    Impl& m = *impl_;
    if (max <= 0) return 0;
    if (m.backend == Backend::Threads) {
        std::unique_lock<std::mutex> lk(m.mu);
        auto ready = [&] { return !m.pool_done.empty() || m.pool_wake; };
        // Nothing queued, nothing being read: waiting forever would never end (the sync path's behaviour).
        if (timeout_ms < 0) m.cv_done.wait(lk, [&] { return ready() || m.pool_pending == 0; });
        else if (timeout_ms > 0) m.cv_done.wait_for(lk, std::chrono::milliseconds(timeout_ms), ready);
        int n = 0;
        while (n < max && !m.pool_done.empty()) {
            out[n++] = m.pool_done.front();
            m.pool_done.pop_front();
        }
        if (n < max && m.pool_wake) {
            m.pool_wake = false;
            out[n++] = Completion{WAKE_TAG, 0, true};
        }
        return n;
    }
#if STRATA_HAVE_IO_URING
    if (m.backend == Backend::Uring) {
        const double deadline = timeout_ms > 0 ? now_us() + 1000.0 * timeout_ms : 0.0;
        for (;;) {
            int n = m.take_done(out, max);
            n += m.reap(out + n, max - n);
            if (n > 0 || timeout_ms == 0) return n;
            if (m.ring_reads.load() == 0 && !m.wake_armed.load()) {
                // Nothing in the kernel: only a fallback read could still show up, and it is already in `done`.
                return m.take_done(out, max);
            }
            if (timeout_ms < 0) {
                if (m.ring.enter(0, 1, IORING_ENTER_GETEVENTS) < 0 && errno != EINTR) return 0;
            } else {
                if (now_us() >= deadline) return 0;
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
    }
#endif
    (void) timeout_ms;
    return m.take_done(out, max);
}
#endif

}  // namespace strata::platform
