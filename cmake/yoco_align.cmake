# Shared by the normal wheel build and component-level incremental builds.
get_filename_component(YOCO_ALIGN_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
define_extension_target(
  _yoco_align_C
  DESTINATION vllm
  LANGUAGE CXX
  SOURCES "${YOCO_ALIGN_SOURCE_ROOT}/csrc/libtorch_stable/yoco_align_gemm.cpp"
  INCLUDE_DIRECTORIES "${YOCO_ALIGN_SOURCE_ROOT}/csrc"
  LIBRARIES CUDA::cudart CUDA::cublasLt
  USE_SABI 3
  WITH_SOABI)
target_compile_features(_yoco_align_C PRIVATE cxx_std_20)
target_compile_definitions(_yoco_align_C PRIVATE
  USE_CUDA TORCH_TARGET_VERSION=0x020B000000000000ULL)
