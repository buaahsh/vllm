#include <torch/all.h>

void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output, bool renormalize,
                  std::optional<torch::Tensor> bias);

TORCH_LIBRARY(yoco_topk_bench, m) {
  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output, bool renormalize, Tensor? "
      "bias) -> ()");
  m.impl("topk_softmax", torch::kCUDA, &topk_softmax);
}
