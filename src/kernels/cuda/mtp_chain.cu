// src/kernels/cuda/mtp_chain.cu - see include/strata/kernels/mtp_chain.hpp.
#include "strata/kernels/mtp_chain.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

__global__ void mtp_chain_init_kernel(int32_t* ctl, const float* __restrict__ probs, const int32_t* __restrict__ row_dev,
                                      const volatile int32_t* params, volatile int32_t* n_out,
                                      cudaGraphConditionalHandle h) {
    const int limit = params[0];
    const float min_p = __int_as_float(params[1]);
    const float p = probs[*row_dev];
    ctl[0] = 1;
    *n_out = 1;
    cudaGraphSetConditional(h, (1 < limit && p >= min_p) ? 1u : 0u);
}

__global__ void mtp_chain_stage_kernel(const int32_t* __restrict__ ctl, const volatile int32_t* step_src,
                                       const volatile int32_t* pos_src, int row0, int n_head, int32_t* step_dst,
                                       int32_t* pos_dst) {
    const int row = row0 + ctl[0] - 1;
    for (int i = threadIdx.x; i < 4; i += blockDim.x) step_dst[i] = step_src[row * 4 + i];
    for (int i = threadIdx.x; i < n_head; i += blockDim.x) pos_dst[i] = pos_src[(size_t) row * n_head + i];
}

__global__ void mtp_chain_next_kernel(int32_t* ctl, const int32_t* __restrict__ ids, const float* __restrict__ probs,
                                      const int32_t* __restrict__ row_dev, volatile int32_t* out, volatile float* out_p,
                                      const volatile int32_t* params, volatile int32_t* n_out,
                                      cudaGraphConditionalHandle h) {
    const int j = ctl[0];
    const int row = *row_dev;
    const float p = probs[row];
    out[j] = ids[row];
    out_p[j] = p;
    const int limit = params[0];
    const float min_p = __int_as_float(params[1]);
    ctl[0] = j + 1;
    *n_out = j + 1;
    cudaGraphSetConditional(h, (j + 1 < limit && p >= min_p) ? 1u : 0u);
}

}  // namespace

void mtp_chain_init(int32_t* ctl, const float* probs, const int32_t* row_dev, const int32_t* params, int32_t* n_out,
                    unsigned long long cond_handle, void* stream) {
    mtp_chain_init_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(ctl, probs, row_dev, (const volatile int32_t*) params,
                                                             (volatile int32_t*) n_out,
                                                             (cudaGraphConditionalHandle) cond_handle);
    check("mtp_chain_init");
}

void mtp_chain_stage(const int32_t* ctl, const int32_t* step_src, const int32_t* pos_src, int row0, int n_head,
                     int32_t* step_dst, int32_t* pos_dst, void* stream) {
    mtp_chain_stage_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(ctl, (const volatile int32_t*) step_src,
                                                               (const volatile int32_t*) pos_src, row0, n_head,
                                                               step_dst, pos_dst);
    check("mtp_chain_stage");
}

void mtp_chain_next(int32_t* ctl, const int32_t* ids, const float* probs, const int32_t* row_dev, int32_t* out,
                    float* out_p, const int32_t* params, int32_t* n_out, unsigned long long cond_handle, void* stream) {
    mtp_chain_next_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(ctl, ids, probs, row_dev, (volatile int32_t*) out,
                                                             (volatile float*) out_p, (const volatile int32_t*) params,
                                                             (volatile int32_t*) n_out,
                                                             (cudaGraphConditionalHandle) cond_handle);
    check("mtp_chain_next");
}

bool build_while_graph(void* stream, const std::function<bool(unsigned long long)>& head,
                       const std::function<bool(unsigned long long)>& body, void** exec_out, std::string& err) {
    cudaStream_t s = (cudaStream_t) stream;
    *exec_out = nullptr;
    auto fail = [&](const char* what, cudaError_t e, cudaGraph_t graph) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("while graph: ") + what + (e != cudaSuccess ? std::string(": ") + cudaGetErrorString(e) : "");
        cudaGetLastError();   // a refused capture leaves a (non-sticky) error behind
        return false;
    };
    cudaGraph_t graph = nullptr;
    cudaError_t e = cudaGraphCreate(&graph, 0);
    if (e != cudaSuccess) return fail("create", e, nullptr);
    cudaGraphConditionalHandle h = 0;
    e = cudaGraphConditionalHandleCreate(&h, graph, 0, cudaGraphCondAssignDefault);
    if (e != cudaSuccess) return fail("conditional handle", e, graph);
    // ---- the head, then the WHILE node after it
    e = cudaStreamBeginCaptureToGraph(s, graph, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) return fail("begin head capture", e, graph);
    bool ok = head((unsigned long long) h);
    cudaGraph_t body_graph = nullptr;
    if (ok) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        cudaGraph_t cap = nullptr;
        const cudaGraphNode_t* deps = nullptr;
        size_t n_deps = 0;
        e = cudaStreamGetCaptureInfo(s, &st, nullptr, &cap, &deps, nullptr, &n_deps);
        if (e != cudaSuccess || st != cudaStreamCaptureStatusActive) ok = false;
        cudaGraphNode_t node = nullptr;
        cudaGraphNodeParams p = {};
        p.type = cudaGraphNodeTypeConditional;
        p.conditional.handle = h;
        p.conditional.type = cudaGraphCondTypeWhile;
        p.conditional.size = 1;
        if (ok && (e = cudaGraphAddNode(&node, cap, deps, nullptr, n_deps, &p)) != cudaSuccess) ok = false;
        if (ok && (e = cudaStreamUpdateCaptureDependencies(s, &node, nullptr, 1, cudaStreamSetCaptureDependencies)) !=
                      cudaSuccess)
            ok = false;
        if (ok) body_graph = p.conditional.phGraph_out[0];
    }
    cudaGraph_t ended = nullptr;
    const cudaError_t ee = cudaStreamEndCapture(s, &ended);
    if (!ok || ee != cudaSuccess) return fail("head capture", ok ? ee : e, graph);
    // ---- the body, into the node's own graph
    e = cudaStreamBeginCaptureToGraph(s, body_graph, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) return fail("begin body capture", e, graph);
    ok = body((unsigned long long) h);
    const cudaError_t be = cudaStreamEndCapture(s, &ended);
    if (!ok) return fail("body recording", cudaSuccess, graph);
    if (be != cudaSuccess) return fail("body capture", be, graph);
    cudaGraphExec_t exec = nullptr;
    e = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    if (e != cudaSuccess) return fail("instantiate", e, nullptr);
    // an explicit upload, as for the other graphs (mtp.cpp: the implicit one blocked behind a device-side spin)
    cudaGraphUpload(exec, s);
    e = cudaStreamSynchronize(s);
    if (e != cudaSuccess) {
        cudaGraphExecDestroy(exec);
        return fail("upload", e, nullptr);
    }
    *exec_out = (void*) exec;
    return true;
}

}  // namespace strata::kernels
