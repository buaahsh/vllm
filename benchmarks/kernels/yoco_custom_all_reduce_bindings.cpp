#include <torch/library.h>
#include <torch/torch.h>

#include <cstdint>
#include <tuple>
#include <vector>

using fptr_t = int64_t;

fptr_t init_custom_ar(const std::vector<int64_t>& fake_ipc_ptrs,
                      torch::Tensor& rank_data, int64_t rank,
                      bool fully_connected);
void all_reduce(fptr_t fa, torch::Tensor& input, torch::Tensor& output,
                fptr_t registered_buffer, int64_t registered_buffer_bytes);
void dispose(fptr_t fa);
int64_t meta_size();
void register_buffer(fptr_t fa, const std::vector<int64_t>& fake_ipc_ptrs);
std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t fa);
void register_graph_buffers(fptr_t fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets);
std::tuple<int64_t, torch::Tensor> allocate_shared_buffer_and_handle(
    int64_t size);
int64_t open_mem_handle(torch::Tensor& mem_handle);
void free_shared_buffer(int64_t buffer);

TORCH_LIBRARY(yoco_custom_ar, ops) {
  ops.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, int rank, "
      "bool fully_connected) -> int");
  ops.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);
  ops.def(
      "all_reduce(int fa, Tensor input, Tensor! output, "
      "int registered_buffer, int registered_buffer_bytes) -> ()");
  ops.impl("all_reduce", torch::kCUDA, &all_reduce);
  ops.def("dispose", &dispose);
  ops.def("meta_size", &meta_size);
  ops.def("register_buffer", &register_buffer);
  ops.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  ops.def("register_graph_buffers", &register_graph_buffers);
  ops.def("allocate_shared_buffer_and_handle",
          &allocate_shared_buffer_and_handle);
  ops.def("open_mem_handle(Tensor mem_handle) -> int");
  ops.impl("open_mem_handle", torch::kCPU, &open_mem_handle);
  ops.def("free_shared_buffer", &free_shared_buffer);
}
