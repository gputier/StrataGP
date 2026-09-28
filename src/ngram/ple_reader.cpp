// src/ngram/ple_reader.cpp - see include/strata/ngram/ple_reader.hpp.
#include "strata/ngram/ple_reader.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace strata::ngram {

using platform::Completion;
using platform::DirectFile;
using platform::now_us;

double ReaderStats::percentile(double q) const {
    if (read_us.empty()) return 0.0;
    std::vector<float> v(read_us);
    const size_t k = std::min(v.size() - 1, (size_t) (q * (double) (v.size() - 1) + 0.5));
    std::nth_element(v.begin(), v.begin() + (ptrdiff_t) k, v.end());
    return v[k];
}

namespace {

constexpr size_t LATENCY_RING = 65536;
constexpr uint32_t WAYS = 8;
constexpr uint32_t EMPTY = 0xFFFFFFFFu;

/// Set-associative row cache: 8 ways per set, round-robin replacement inside a set. Bounded by construction.
struct RowCache {
    uint64_t sets = 0;
    std::vector<uint32_t> keys;     // sets * WAYS
    std::vector<uint8_t> data;      // sets * WAYS * ROW_BYTES
    std::vector<uint8_t> next;      // per-set replacement pointer
    uint64_t used = 0;

    void init(uint64_t rows) {
        sets = rows / WAYS;
        keys.assign(sets * WAYS, EMPTY);
        data.assign(sets * WAYS * ROW_BYTES, 0);
        next.assign(sets, 0);
        used = 0;
    }
    static uint64_t mix(uint32_t r) {
        uint64_t x = r * 0x9E3779B97F4A7C15ull;
        return x ^ (x >> 29);
    }
    const uint8_t* find(uint32_t row) const {
        if (sets == 0) return nullptr;
        const uint64_t s = mix(row) % sets;
        for (uint32_t w = 0; w < WAYS; ++w)
            if (keys[s * WAYS + w] == row) return &data[(s * WAYS + w) * ROW_BYTES];
        return nullptr;
    }
    void insert(uint32_t row, const uint8_t* bytes) {
        if (sets == 0 || find(row) != nullptr) return;
        const uint64_t s = mix(row) % sets;
        uint32_t w = next[s];
        next[s] = (uint8_t) ((w + 1) % WAYS);
        if (keys[s * WAYS + w] == EMPTY) ++used;
        keys[s * WAYS + w] = row;
        std::memcpy(&data[(s * WAYS + w) * ROW_BYTES], bytes, ROW_BYTES);
    }
};

struct Use {
    uint32_t row;
    uint32_t in_page;      // byte offset of the row inside the read buffer
    uint8_t* dst;
};

struct Job {
    uint64_t offset = 0;   // aligned file offset
    uint32_t length = 0;   // PAGE or 2 * PAGE (a row that straddles a page boundary)
    uint32_t ticket = 0;
    std::vector<Use> uses;
    double issued_us = 0;
};

struct TicketState {
    uint32_t pending = 0;  // jobs not yet completed
};

}  // namespace

// THREADING. With `io_thread` (the default) one worker thread owns every DirectFile call: it submits queued jobs
// and reaps completions, so `issue` on the caller's thread only builds jobs and wakes it (~11 us per ReadFile
// no longer lands on the host loop). `mu` guards everything below except `file`, which only the worker touches
// (plus `wake`, which is thread-safe), and the `inflight` jobs, which only the worker touches while the reader is
// open. Issue #15: the worker makes its submit calls WITHOUT `mu` (`pump_worker`), so a submit never holds up
// `issue` or `collect`. The `sync` backend keeps the previous locked pump (`pump_locked`: pread under `mu`, and
// `finish` refills the freed slot at once), so it stays a faithful A/B arm. Without `io_thread` the caller does all
// of it.
struct PleReader::Impl {
    DirectFile file;
    uint64_t table_offset = 0;
    uint64_t n_rows = 0;
    uint32_t max_inflight = 0;
    uint8_t* slab = nullptr;              // max_inflight slots of 2 pages
    std::vector<uint32_t> free_slots;
    std::vector<Job> inflight;            // indexed by slot
    std::vector<Completion> delayed;      // completed but held back by fault injection
    std::deque<Job> queue;                // not yet submitted
    std::vector<uint32_t> batch;          // pump_worker: the slots taken in one pass
    std::unordered_map<uint32_t, TicketState> tickets;
    uint32_t next_ticket = 1;
    RowCache cache;
    ReaderStats stats;
    size_t ring_pos = 0;
    double delay_us = 0;
    std::string error;

    bool threaded = false;
    bool pump_locked = false;             // the worker submits under `mu` (the `sync` backend, as before issue #15)
    bool stop = false;
    std::mutex mu;
    std::condition_variable cv_work;      // worker: there is something to submit
    std::condition_variable cv_done;      // collectors: a ticket may have completed
    std::thread worker;

    uint8_t* slot_buf(uint32_t s) { return slab + (size_t) s * 2 * PAGE; }
    bool busy() const { return free_slots.size() < max_inflight || !delayed.empty(); }

    void record_latency(double us) {
        stats.read_us_sum += us;
        if (stats.read_us.size() < LATENCY_RING) stats.read_us.push_back((float) us);
        else stats.read_us[ring_pos++ % LATENCY_RING] = (float) us;
    }

    bool pump() {
        while (!queue.empty() && !free_slots.empty()) {
            const uint32_t s = free_slots.back();
            free_slots.pop_back();
            inflight[s] = std::move(queue.front());
            queue.pop_front();
            Job& j = inflight[s];
            j.issued_us = now_us();
            if (!file.submit(j.offset, slot_buf(s), j.length, s, error)) {
                j.uses.clear();                    // not queued: no completion will free the slot
                free_slots.push_back(s);
                return false;
            }
            stats.submit_us += now_us() - j.issued_us;
            ++stats.reads;
        }
        return true;
    }

    /// The worker's pump: slots are taken under `mu` (held by `lk` on entry and on return), the submit calls run
    /// without it. A job that could not be submitted gives its slot back; its ticket then fails through `error`.
    bool pump_worker(std::unique_lock<std::mutex>& lk) {
        batch.clear();
        while (!queue.empty() && !free_slots.empty()) {
            const uint32_t s = free_slots.back();
            free_slots.pop_back();
            inflight[s] = std::move(queue.front());
            queue.pop_front();
            batch.push_back(s);
        }
        if (batch.empty()) return true;
        lk.unlock();
        std::string e;
        size_t done = 0;
        double sub_us = 0;
        for (; done < batch.size(); ++done) {
            Job& j = inflight[batch[done]];
            j.issued_us = now_us();
            if (!file.submit(j.offset, slot_buf(batch[done]), j.length, batch[done], e)) break;
            sub_us += now_us() - j.issued_us;
        }
        lk.lock();
        stats.submit_us += sub_us;
        stats.reads += done;
        if (done == batch.size()) return true;
        for (size_t i = done; i < batch.size(); ++i) {
            inflight[batch[i]].uses.clear();
            free_slots.push_back(batch[i]);
        }
        error = e.empty() ? "PleReader: submit failed" : e;
        return false;
    }

    /// One completed read. After an error (this read's or an earlier one) nothing is copied - every `collect`
    /// fails from then on, and a failed ticket's destination may already be gone - but the slot is still given
    /// back, so the reader drains and `close` returns.
    bool finish(const Completion& c) {
        const uint32_t s = (uint32_t) c.tag;
        Job& j = inflight[s];
        bool ok = error.empty();
        if (ok && !c.ok) {
            error = "PleReader: a table read failed";
            ok = false;
        }
        if (ok) {
            record_latency(now_us() - j.issued_us);
            stats.bytes += c.bytes;
            const uint8_t* buf = slot_buf(s);
            for (const Use& u : j.uses) {
                if (u.in_page + ROW_BYTES > c.bytes) {
                    error = "PleReader: short read inside the table";
                    ok = false;
                    break;
                }
                std::memcpy(u.dst, buf + u.in_page, ROW_BYTES);
                cache.insert(u.row, buf + u.in_page);
            }
        }
        if (!ok) {
            j.uses.clear();
            free_slots.push_back(s);
            return false;
        }
        auto it = tickets.find(j.ticket);
        if (it != tickets.end() && it->second.pending > 0) --it->second.pending;
        j.uses.clear();
        free_slots.push_back(s);
        return (threaded && !pump_locked) || pump();   // else the worker pumps at the top of its loop, without `mu`
    }

    /// Completions (or wake packets) just returned by `file.wait`, applying fault injection.
    /// Every completion is finished even after one fails, so no slot is lost.
    bool process(const Completion* got, int n) {
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (got[i].tag == DirectFile::WAKE_TAG) continue;
            if (delay_us > 0 && now_us() - inflight[(uint32_t) got[i].tag].issued_us < delay_us) {
                delayed.push_back(got[i]);
                ++stats.late_injected;
                continue;
            }
            if (!finish(got[i])) ok = false;
        }
        return ok;
    }

    bool release_delayed() {
        const double now = now_us();
        bool ok = true;
        for (size_t i = 0; i < delayed.size();) {
            if (now - inflight[(uint32_t) delayed[i].tag].issued_us >= delay_us) {
                const Completion c = delayed[i];
                delayed.erase(delayed.begin() + (ptrdiff_t) i);
                if (!finish(c)) ok = false;
            } else {
                ++i;
            }
        }
        return ok;
    }

    /// Caller-thread mode: process whatever has completed; blocks up to `timeout_ms` for the first completion.
    bool drain(int timeout_ms) {
        Completion got[64];
        if (!delayed.empty()) {
            if (!release_delayed()) return false;
            timeout_ms = 0;                        // keep polling the held completions
        }
        const int n = file.wait(got, 64, timeout_ms);
        return process(got, n);
    }

    void worker_loop() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv_work.wait(lk, [&] { return stop || !queue.empty() || busy(); });
            if (stop && !busy()) break;
            // Held completions first, so the pump right after refills the slots they free (with `pump_worker`,
            // `finish` does not pump: a freed slot would otherwise wait for the next completion).
            if (!delayed.empty() && !release_delayed() && error.empty()) error = "PleReader: read failed";
            if (error.empty() && !(pump_locked ? pump() : pump_worker(lk)) && error.empty())
                error = "PleReader: submit failed";
            if (!error.empty()) {
                cv_done.notify_all();
                if (!busy()) { cv_work.wait(lk, [&] { return stop; }); break; }
            }
            if (!busy()) { cv_done.notify_all(); continue; }
            // Block in the port WITHOUT the lock, so `issue` can queue work; `issue` wakes us with a packet.
            const int timeout = delayed.empty() ? -1 : 0;
            lk.unlock();
            Completion got[64];
            const int n = file.wait(got, 64, timeout);
            lk.lock();
            if (!process(got, n) && error.empty()) error = "PleReader: read failed";
            cv_done.notify_all();
        }
    }
};

PleReader::PleReader() : impl_(new Impl) {}
PleReader::~PleReader() {
    close();
    delete impl_;
}

bool PleReader::open(const std::string& path, uint64_t table_offset, uint64_t n_rows, uint32_t max_inflight,
                     uint64_t cache_rows, std::string& err, bool io_thread) {
    close();
    if (max_inflight == 0 || max_inflight > 1024) { err = "PleReader: max_inflight must be 1..1024"; return false; }
    if (!impl_->file.open(path, err, max_inflight)) return false;
    if (table_offset + n_rows * (uint64_t) ROW_BYTES > impl_->file.size()) {
        err = "PleReader: the table extends past the end of " + path;
        close();
        return false;
    }
    impl_->table_offset = table_offset;
    impl_->n_rows = n_rows;
    impl_->max_inflight = max_inflight;
    impl_->slab = (uint8_t*) DirectFile::alloc_aligned((size_t) max_inflight * 2 * PAGE);
    if (impl_->slab == nullptr) { err = "PleReader: cannot allocate read buffers"; close(); return false; }
    impl_->inflight.assign(max_inflight, Job{});
    impl_->free_slots.clear();
    for (uint32_t s = max_inflight; s-- > 0;) impl_->free_slots.push_back(s);
    impl_->cache.init(cache_rows);
    impl_->error.clear();
    reset_stats();
    impl_->stop = false;
    impl_->threaded = io_thread;
    impl_->pump_locked = io_thread && std::strcmp(impl_->file.backend(), "sync") == 0;
    if (io_thread) impl_->worker = std::thread([this] { impl_->worker_loop(); });
    return true;
}

void PleReader::close() {
    Impl& m = *impl_;
    if (m.worker.joinable()) {
        {
            std::lock_guard<std::mutex> lk(m.mu);
            m.stop = true;
            m.queue.clear();                       // unsubmitted work is dropped; in-flight reads still drain
        }
        m.cv_work.notify_all();
        m.file.wake();
        m.worker.join();
    }
    // Outstanding reads must finish before their buffers are released (caller-thread mode, or a worker that
    // stopped on an error).
    if (m.file.is_open()) {
        while (m.free_slots.size() < m.max_inflight) {
            Completion c[64];
            const int n = m.file.wait(c, 64, -1);
            if (n == 0) break;
            for (int i = 0; i < n; ++i)
                if (c[i].tag != DirectFile::WAKE_TAG) m.free_slots.push_back((uint32_t) c[i].tag);
        }
    }
    m.file.close();
    DirectFile::free_aligned(m.slab);
    m.slab = nullptr;
    m.queue.clear();
    m.tickets.clear();
    m.delayed.clear();
    m.inflight.clear();
    m.free_slots.clear();
    m.cache.init(0);
    m.threaded = false;
    m.pump_locked = false;
    m.stop = false;
}

bool PleReader::is_open() const { return impl_->file.is_open(); }

PleReader::Ticket PleReader::issue(const uint32_t* rows, size_t n, uint8_t* out_raw) {
    Impl& m = *impl_;
    std::unique_lock<std::mutex> lk(m.mu, std::defer_lock);
    if (m.threaded) lk.lock();
    const uint32_t id = m.next_ticket++;
    if (m.next_ticket == 0) m.next_ticket = 1;
    TicketState& ts = m.tickets[id];
    std::unordered_map<uint64_t, size_t> by_page;     // aligned offset -> index in `jobs`
    std::vector<Job> jobs;
    for (size_t i = 0; i < n; ++i) {
        uint8_t* dst = out_raw + i * ROW_BYTES;
        ++m.stats.requests;
        if (rows[i] >= m.n_rows) {
            std::memset(dst, 0, ROW_BYTES);
            continue;
        }
        if (const uint8_t* hit = m.cache.find(rows[i])) {
            std::memcpy(dst, hit, ROW_BYTES);
            ++m.stats.cache_hits;
            continue;
        }
        const uint64_t at = m.table_offset + (uint64_t) rows[i] * ROW_BYTES;
        const uint64_t first = at / PAGE * PAGE;
        const uint32_t length = (uint32_t) ((at + ROW_BYTES - 1) / PAGE * PAGE - first + PAGE);
        auto f = by_page.find(first);
        if (f != by_page.end()) {
            Job& j = jobs[f->second];
            j.length = std::max(j.length, length);
            j.uses.push_back(Use{rows[i], (uint32_t) (at - first), dst});
            ++m.stats.dedup_rows;
            continue;
        }
        by_page.emplace(first, jobs.size());
        Job j;
        j.offset = first;
        j.length = length;
        j.ticket = id;
        j.uses.push_back(Use{rows[i], (uint32_t) (at - first), dst});
        jobs.push_back(std::move(j));
    }
    // Sorted by offset: prefill chunks then read the SSD in near-sequential order.
    std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) { return a.offset < b.offset; });
    ts.pending = (uint32_t) jobs.size();
    for (Job& j : jobs) m.queue.push_back(std::move(j));
    if (m.threaded) {
        const bool queued = ts.pending > 0;        // read under `mu`: the worker updates it once unlocked
        lk.unlock();
        if (queued) {
            m.cv_work.notify_one();
            m.file.wake();                         // in case the worker is blocked in the port
        }
    } else if (!m.pump() && m.error.empty()) {
        m.error = "PleReader: submit failed";
    }
    return Ticket{id};
}

bool PleReader::collect(Ticket t, std::string& err) {
    Impl& m = *impl_;
    const double start = now_us();
    if (m.threaded) {
        std::unique_lock<std::mutex> lk(m.mu);
        auto it = m.tickets.find(t.id);
        if (it == m.tickets.end()) { err = "PleReader: unknown ticket"; return false; }
        m.cv_done.wait(lk, [&] { return !m.error.empty() || m.tickets[t.id].pending == 0; });
        if (!m.error.empty()) { err = m.error; return false; }
        m.stats.wait_us += now_us() - start;
        m.tickets.erase(t.id);
        return true;
    }
    auto it = m.tickets.find(t.id);
    if (it == m.tickets.end()) { err = "PleReader: unknown ticket"; return false; }
    while (it->second.pending > 0) {
        if (!m.error.empty() || !m.drain(-1)) {
            err = m.error.empty() ? "PleReader: read failed" : m.error;
            return false;
        }
        it = m.tickets.find(t.id);
    }
    m.stats.wait_us += now_us() - start;
    m.tickets.erase(it);
    return true;
}

void PleReader::set_injected_delay_us(double delay_us) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->delay_us = delay_us < 0 ? 0 : delay_us;
}
const ReaderStats& PleReader::stats() const { return impl_->stats; }
void PleReader::reset_stats() {
    impl_->stats = ReaderStats{};
    impl_->ring_pos = 0;
}
const char* PleReader::backend() const { return impl_->file.backend(); }
uint64_t PleReader::cache_capacity() const { return impl_->cache.sets * WAYS; }
uint64_t PleReader::cache_size() const { return impl_->cache.used; }

}  // namespace strata::ngram
