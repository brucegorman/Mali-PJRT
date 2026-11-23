cmake_minimum_required(VERSION 3.22)
project(jax_mali_probe LANGUAGES C)

if(NOT ANDROID)
  message(FATAL_ERROR "Configure with the Android NDK toolchain")
endif()

add_executable(vulkan_probe vulkan_probe.c)
target_compile_features(vulkan_probe PRIVATE c_std_11)
target_compile_options(vulkan_probe PRIVATE -Wall -Wextra -Werror)
target_link_libraries(vulkan_probe PRIVATE vulkan)
