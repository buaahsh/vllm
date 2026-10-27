// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cublasLt.h>
#include <torch/csrc/stable/library.h>

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <tuple>

#include "torch_utils.h"
#include "../core/registration.h"

namespace {
using torch::stable::Tensor;

void check(cublasStatus_t status) {
  STD_TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                  "YOCO Align cuBLASLt status ", static_cast<int>(status));
}

// Handles/descriptors belong to this provider, never to ATen's BLAS pool.
struct Handle {
  cublasLtHandle_t value = nullptr;
  Handle() { check(cublasLtCreate(&value)); }
  ~Handle() { cublasLtDestroy(value); }
};

struct Plan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulAlgo_t algo{};
  size_t workspace = 0;
  int id = -1;

  ~Plan() {
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (op) cublasLtMatmulDescDestroy(op);
  }
};

uint32_t alignment(const void* ptr) {
  auto address = reinterpret_cast<uintptr_t>(ptr);
  uint32_t value = 256;
  while (address % value) value /= 2;
  return value;
}

// Alignment and leading dimensions are part of the algorithm contract.
using Key = std::tuple<int, int64_t, int64_t, int64_t, int64_t, int64_t,
                       int64_t, uint32_t, uint32_t>;
struct State {
  Handle handle;
  std::map<Key, std::unique_ptr<Plan>> plans;
};

void validate(const Tensor& x, const Tensor& w) {
  STD_TORCH_CHECK(
      x.device().is_cuda() && w.device().is_cuda() && x.device() == w.device(),
      "YOCO Align GEMM requires tensors on the same CUDA device");
  STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
                      w.scalar_type() == x.scalar_type(),
                  "YOCO Align GEMM requires BF16 tensors");
  STD_TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1) &&
                      x.size(0) > 0 && w.size(0) > 0 && x.size(1) > 0,
                  "YOCO Align GEMM expects nonempty [M,K] and [N,K]");
  STD_TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 &&
                      x.stride(0) >= x.size(1) && w.stride(0) >= w.size(1),
                  "YOCO Align GEMM requires row-major operands");
}

Plan& get_plan(State& state, const Tensor& x, const Tensor& w,
               int64_t workspace_bytes) {
  STD_TORCH_CHECK(workspace_bytes >= 0, "Negative workspace budget");
  const int64_t m = x.size(0), n = w.size(0), k = x.size(1);
  const auto ax = alignment(x.data_ptr()), aw = alignment(w.data_ptr());
  const Key key{x.get_device_index(), m,  n, k, x.stride(0), w.stride(0),
                workspace_bytes,      ax, aw};
  auto found = state.plans.find(key);
  if (found != state.plans.end()) return *found->second;

  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  STD_TORCH_CHECK(
      cudaStreamIsCapturing(get_current_cuda_stream(x.get_device_index()),
                            &capture) == cudaSuccess,
      "Cannot query YOCO Align stream capture state");
  STD_TORCH_CHECK(capture == cudaStreamCaptureStatusNone,
                  "Warm up YOCO Align GEMM shapes before CUDA Graph capture");

  // Host-only selection is cached by shape. No autotuning, synchronization,
  // or scratch allocation is performed by the selector during Graph capture.
  auto plan = std::make_unique<Plan>();
  check(cublasLtMatmulDescCreate(&plan->op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t transa = CUBLAS_OP_T;
  check(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &transa, sizeof(transa)));
  // Row-major X @ W.T is column-major W @ X.T: output layout [N,M].
  check(cublasLtMatrixLayoutCreate(&plan->a, CUDA_R_16BF, k, n, w.stride(0)));
  check(cublasLtMatrixLayoutCreate(&plan->b, CUDA_R_16BF, k, m, x.stride(0)));
  check(cublasLtMatrixLayoutCreate(&plan->c, CUDA_R_16BF, n, m, n));
  check(cublasLtMatmulPreferenceCreate(&plan->preference));
  const size_t budget = workspace_bytes;
  check(cublasLtMatmulPreferenceSetAttribute(
      plan->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &budget,
      sizeof(budget)));
  const uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE;
  check(cublasLtMatmulPreferenceSetAttribute(
      plan->preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction,
      sizeof(reduction)));
  check(cublasLtMatmulPreferenceSetAttribute(
      plan->preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &aw,
      sizeof(aw)));
  check(cublasLtMatmulPreferenceSetAttribute(
      plan->preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &ax,
      sizeof(ax)));
  const uint32_t output_alignment = 256;  // CUDA tensor allocator contract.
  for (auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
    check(cublasLtMatmulPreferenceSetAttribute(
        plan->preference, attr, &output_alignment, sizeof(output_alignment)));
  }
  std::array<cublasLtMatmulHeuristicResult_t, 32> candidates{};
  int count = 0;
  check(cublasLtMatmulAlgoGetHeuristic(
      state.handle.value, plan->op, plan->a, plan->b, plan->c, plan->c,
      plan->preference, candidates.size(), candidates.data(), &count));
  bool selected = false;
  for (const auto& candidate : candidates) {
    if (count-- <= 0) break;
    if (candidate.state != CUBLAS_STATUS_SUCCESS ||
        candidate.workspaceSize > budget)
      continue;
    uint32_t split_k = 0, scheme = 0;
    size_t written = 0;
    check(cublasLtMatmulAlgoConfigGetAttribute(
        &candidate.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split_k,
        sizeof(split_k), &written));
    check(cublasLtMatmulAlgoConfigGetAttribute(
        &candidate.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &scheme,
        sizeof(scheme), &written));
    if (split_k > 1 || scheme != CUBLASLT_REDUCTION_SCHEME_NONE) continue;
    plan->algo = candidate.algo;
    plan->workspace = candidate.workspaceSize;
    check(cublasLtMatmulAlgoConfigGetAttribute(
        &plan->algo, CUBLASLT_ALGO_CONFIG_ID, &plan->id, sizeof(plan->id),
        &written));
    selected = true;
    break;
  }
  STD_TORCH_CHECK(selected, "No no-split-K YOCO Align cuBLASLt algorithm for ",
                  m, "x", n, "x", k, ", workspace=", budget);
  auto& result = *plan;
  state.plans.emplace(key, std::move(plan));
  return result;
}

State& state_for(int device) {
  // Separate descriptors/handles across host threads and devices. Workspace
  // is a per-call tensor, so concurrent streams and Graphs never alias it.
  thread_local std::map<int, std::unique_ptr<State>> states;
  auto& state = states[device];
  if (!state) state = std::make_unique<State>();
  return *state;
}

Tensor yoco_align_gemm(const Tensor& x, const Tensor& w, int64_t budget) {
  validate(x, w);
  const torch::stable::accelerator::DeviceGuard guard(x.get_device_index());
  auto& state = state_for(x.get_device_index());
  auto& plan = get_plan(state, x, w, budget);
  auto output = torch::stable::empty({x.size(0), w.size(0)}, x.scalar_type(),
                                     std::nullopt, x.device());
  auto workspace = torch::stable::empty({static_cast<int64_t>(plan.workspace)},
                                        torch::headeronly::ScalarType::Byte,
                                        std::nullopt, x.device());
  const float alpha = 1.0f, beta = 0.0f;
  check(cublasLtMatmul(state.handle.value, plan.op, &alpha, w.data_ptr(),
                       plan.a, x.data_ptr(), plan.b, &beta, output.data_ptr(),
                       plan.c, output.mutable_data_ptr(), plan.c, &plan.algo,
                       workspace.mutable_data_ptr(), plan.workspace,
                       get_current_cuda_stream(x.get_device_index())));
  return output;
}

std::vector<int64_t> yoco_align_gemm_plan(const Tensor& x, const Tensor& w,
                                          int64_t budget) {
  validate(x, w);
  const torch::stable::accelerator::DeviceGuard guard(x.get_device_index());
  auto& state = state_for(x.get_device_index());
  auto& plan = get_plan(state, x, w, budget);
  std::vector<int64_t> result{static_cast<int64_t>(cublasLtGetVersion()),
                              plan.id, static_cast<int64_t>(plan.workspace)};
  for (auto attr :
       {CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_STAGES_ID,
        CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION}) {
    uint32_t value = 0;
    size_t written = 0;
    check(cublasLtMatmulAlgoConfigGetAttribute(&plan.algo, attr, &value,
                                               sizeof(value), &written));
    result.push_back(value);
  }
  return result;
}

int64_t yoco_align_gemm_version() { return cublasLtGetVersion(); }
}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "yoco_align_gemm(Tensor x, Tensor weight, int workspace_bytes) -> "
      "Tensor");
  ops.def(
      "yoco_align_gemm_plan(Tensor x, Tensor weight, int workspace_bytes) -> "
      "int[]");
  ops.def("yoco_align_gemm_version() -> int");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CompositeExplicitAutograd, ops) {
  ops.impl("yoco_align_gemm_version", TORCH_BOX(&yoco_align_gemm_version));
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("yoco_align_gemm", TORCH_BOX(&yoco_align_gemm));
  ops.impl("yoco_align_gemm_plan", TORCH_BOX(&yoco_align_gemm_plan));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
