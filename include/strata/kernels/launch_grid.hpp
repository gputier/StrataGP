// include/strata/kernels/launch_grid.hpp - issue #26 (O7): decode grids sized from the device, not for 48 SMs.
//
// Several decode launches were written for the 48 SMs of the RTX 5070: the blob and row copies ran on `48 * 8`
// blocks, the GDN alpha/beta projection on 12 blocks (a warp per row) and the GDN step on one block per value head
// (48).  On a 170-SM RTX 5090 they leave most of the chip idle.  The launchers now read each device once, here.
//
// Every change behind these switches is bitwise: the same values reach the same arithmetic in the same order; only
// the number of blocks, and which warp or block carries a chain, differ.  The previous launches stay selectable, one
// switch per item, for an A/B on the GPU (all three together = 0.1.20):
//
//   STRATA_OLD_GRIDS=1      the grid-stride copies (fetch_blobs, gather_rows) on 48 * 8 blocks again
//   STRATA_OLD_GDN_AB=1     the alpha/beta projection (fused_gdn_ab, gdn_ab_multi) on a warp per row again
//   STRATA_OLD_GDN_STEP=1   the GDN step (fused_gdn_step_norm, gdn_step_norm_multi) on a block per head again
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstdlib>
#include <mutex>

namespace strata::kernels {

/// What the grids are sized from, read once per device (a layer split drives two or three GPUs: see PerDevice).
struct GridDevice {
    int sms = 48;             ///< multiprocessors; 48 (the 0.1.20 grids) when the query fails
    int cc_major = 0;         ///< compute capability, major
    bool clusters = false;    ///< thread block clusters and distributed shared memory (sm_90 and newer)
};

/// Runs `fn` with this thread's stream-capture mode relaxed: the first launch of a kernel may happen inside a
/// capture, and the one-time device queries must neither fail nor invalidate it.
template <typename Fn>
void outside_capture_rules(Fn&& fn) {
    const cudaError_t pending = cudaPeekAtLastError();
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    const bool swapped = cudaThreadExchangeStreamCaptureMode(&mode) == cudaSuccess;
    fn();
    // a failed query must not surface as the next launch's error (an error already pending is left to its owner)
    if (pending == cudaSuccess) (void) cudaGetLastError();
    if (swapped) cudaThreadExchangeStreamCaptureMode(&mode);
}

/// The devices a per-device answer is kept for; a larger ordinal shares the last entry.
constexpr int kGridMaxDevices = 64;

/// The current device's ordinal, clamped to the per-device tables (0 when the query fails).
inline int grid_device_index() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return 0;
    }
    return dev < 0 ? 0 : dev >= kGridMaxDevices ? kGridMaxDevices - 1 : dev;
}

/// A one-time answer kept PER DEVICE: a layer split (docs/MULTI_GPU.md) runs the same launchers on two or three
/// cards, e.g. an RTX 5080 (84 SMs, clusters) beside an RTX 3090 (82 SMs, no clusters), so an answer read once per
/// process would size one card's grids from the other and launch cluster kernels where they cannot run.
template <typename T>
class PerDevice {
public:
    template <typename Fn>
    const T& get(Fn&& fn) {
        const int d = grid_device_index();
        std::call_once(once_[d], [&] { value_[d] = fn(); });
        return value_[d];
    }

private:
    std::once_flag once_[kGridMaxDevices];
    T value_[kGridMaxDevices] = {};
};

inline const GridDevice& grid_device() {
    static PerDevice<GridDevice> devices;
    return devices.get([] {
        GridDevice d;
        outside_capture_rules([&] {
            int dev = 0, v = 0;
            if (cudaGetDevice(&dev) != cudaSuccess) return;
            if (cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && v > 0) d.sms = v;
            if (cudaDeviceGetAttribute(&v, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess) d.cc_major = v;
            if (cudaDeviceGetAttribute(&v, cudaDevAttrClusterLaunch, dev) == cudaSuccess)
                d.clusters = v != 0 && d.cc_major >= 9;
        });
        return d;
    });
}

/// A `STRATA_OLD_*` switch: read from the environment once; `set` overrides it (the parity test runs both paths).
class OldPath {
public:
    explicit OldPath(const char* name) {
        const char* v = std::getenv(name);
        on_.store(v != nullptr && v[0] != '\0' && v[0] != '0', std::memory_order_relaxed);
    }
    bool on() const { return on_.load(std::memory_order_relaxed); }
    void set(bool v) { on_.store(v, std::memory_order_relaxed); }

private:
    std::atomic<bool> on_{false};
};

inline OldPath& old_grids() { static OldPath p("STRATA_OLD_GRIDS"); return p; }
inline OldPath& old_gdn_ab() { static OldPath p("STRATA_OLD_GDN_AB"); return p; }
inline OldPath& old_gdn_step() { static OldPath p("STRATA_OLD_GDN_STEP"); return p; }

/// The multiprocessor count a grid-stride launch is sized for: the device's, or 48 under STRATA_OLD_GRIDS=1.
inline int grid_sms() { return old_grids().on() ? 48 : grid_device().sms; }

/// Whether `kernel` can be launched here as clusters of `cluster` blocks of `block` threads.  The kernel must have
/// been compiled for sm_90 or newer: in a build for an older architecture its body is an empty stub, and this says
/// no.  Called once per kernel and device (the caller keeps the answer in a PerDevice).
template <typename Kernel>
bool cluster_launchable(Kernel* kernel, dim3 block, unsigned cluster) {
    if (!grid_device().clusters) return false;
    bool ok = false;
    outside_capture_rules([&] {
        cudaFuncAttributes fa{};
        if (cudaFuncGetAttributes(&fa, kernel) != cudaSuccess || fa.ptxVersion < 90) return;
        cudaLaunchAttribute at{};
        at.id = cudaLaunchAttributeClusterDimension;
        at.val.clusterDim.x = cluster;
        at.val.clusterDim.y = 1;
        at.val.clusterDim.z = 1;
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(cluster);
        cfg.blockDim = block;
        cfg.attrs = &at;
        cfg.numAttrs = 1;
        int n = 0;
        ok = cudaOccupancyMaxActiveClusters(&n, kernel, &cfg) == cudaSuccess && n > 0;
    });
    return ok;
}

/// Whether the GDN step runs on a cluster of blocks per head (false: a device or a build without clusters, or an
/// old-path switch).  Defined in fused_gdn.cu and verify_kernels.cu; the parity test prints them.
bool fused_gdn_step_cluster_path();
bool gdn_step_norm_multi_cluster_path();

}  // namespace strata::kernels
