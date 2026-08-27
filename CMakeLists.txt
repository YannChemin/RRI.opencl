cmake_minimum_required(VERSION 3.16)
project(rri_opencl C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O2")

find_package(OpenMP)
find_package(OpenCL)

set(RRI_CORE_SRCS
  src/rri_io.c
  src/rri_setup.c
  src/rri_rk.c
  src/rri_riv.c
  src/rri_slope.c
  src/rri_gw.c
  src/rri_infilt.c
  src/rri_rivslo.c
)

add_library(rri_core STATIC ${RRI_CORE_SRCS})
target_include_directories(rri_core PUBLIC include)
target_link_libraries(rri_core PUBLIC m)
if(OpenMP_C_FOUND)
  target_link_libraries(rri_core PUBLIC OpenMP::OpenMP_C)
endif()

add_executable(rri_cpu src/main.c)
target_link_libraries(rri_cpu PRIVATE rri_core)

enable_testing()
add_subdirectory(tests)

# OpenCL backend: not yet implemented in this checkpoint (see README /
# PLAN.md milestone 8) -- OpenMP-only core is validated first per
# PLAN.md's "prioritize a correct, validated core over a shallow complete
# port" guidance. find_package(OpenCL) above is kept so a future pass can
# wire cl/*.cl kernels + a host dispatch in without restructuring this file.
if(OpenCL_FOUND)
  message(STATUS "OpenCL found (${OpenCL_LIBRARIES}) -- backend not implemented yet, see PLAN.md milestone 8")
endif()
