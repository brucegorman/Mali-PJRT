add_executable(mali_worker
  "${CMAKE_CURRENT_LIST_DIR}/worker.c"
  "${CMAKE_CURRENT_LIST_DIR}/worker_buffers.c"
  "${CMAKE_CURRENT_LIST_DIR}/output_metadata.c"
)
target_compile_features(mali_worker PRIVATE c_std_11)
target_compile_options(mali_worker PRIVATE -Wall -Wextra -Werror)
target_include_directories(mali_worker PRIVATE "${MALI_PJRT_HEADER_DIR}")
target_include_directories(mali_worker SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/runtime/src")
target_link_libraries(mali_worker PRIVATE iree::runtime::unified vulkan)
set_target_properties(mali_worker PROPERTIES
  LINKER_LANGUAGE CXX
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools"
)
