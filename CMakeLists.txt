cmake_minimum_required(VERSION 3.20)
project(NBody-kokkos)

set(Kokkos_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../dep/Kokkos)
find_package(Kokkos CONFIG)
if(Kokkos_FOUND)
  message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
else()
  if(EXISTS ${Kokkos_COMMON_SOURCE_DIR})
    add_subdirectory(${Kokkos_COMMON_SOURCE_DIR} Kokkos)
  else()
    include(FetchContent)
    FetchContent_Declare(
      Kokkos
      GIT_REPOSITORY https://github.com/kokkos/kokkos.git
      GIT_TAG        4.0.01
      SOURCE_DIR ${Kokkos_COMMON_SOURCE_DIR}
    )
    FetchContent_MakeAvailable(Kokkos)
  endif()
endif()



add_executable(NBodyAOS main_aos.cpp)
add_executable(NBodySOA main_soa.cpp)
target_link_libraries(NBodyAOS Kokkos::kokkos)
target_link_libraries(NBodySOA Kokkos::kokkos)

