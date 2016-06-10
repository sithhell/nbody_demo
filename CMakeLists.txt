project(nbody_demo)

cmake_minimum_required(VERSION 2.8)

################################################################################
# enable solution folders for MSVC
if(MSVC)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()
################################################################################

find_package(HPX REQUIRED)
find_package(libgeodecomp REQUIRED)

set(LOCAL_LIBGEODECOMP_LINK_LIB geodecomp)

include_directories(.)
include_directories(${HPX_INCLUDE_DIR})
include_directories(${libgeodecomp_INCLUDE_DIRS})

add_definitions(-DQT_NO_KEYWORDS)
add_definitions(${HPX_CXX_COMPILE_FLAGS})

add_hpx_executable(
    nbody_smp
    ESSENTIAL
    SOURCES main.cpp
    HEADERS
        interactor_scalar.hpp
        interactor_scalar_swapped.hpp
        interactor_sse.hpp
        interactor_sse_swapped.hpp
        nbody_container.hpp
        nbody_initializer.hpp
        run_simulation.hpp
    COMPILE_FLAGS -fopenmp
    LINK_FLAGS -fopenmp
    DEPENDENCIES
        geodecomp
    FOLDER "nbody SMP"
)

add_hpx_executable(
    nbody_mpi_weak
    ESSENTIAL
    SOURCES main_weak.cpp
    HEADERS
        interactor_scalar.hpp
        interactor_scalar_swapped.hpp
        interactor_sse.hpp
        interactor_sse_swapped.hpp
        nbody_container.hpp
        nbody_initializer.hpp
        run_simulation.hpp
    COMPILE_FLAGS -fopenmp
    LINK_FLAGS -fopenmp
    DEPENDENCIES
        geodecomp
    FOLDER "nbody MPI Weak"
)

add_hpx_executable(
    nbody_hpx_weak
    ESSENTIAL
    SOURCES main_weak.cpp
    HEADERS
        interactor_scalar.hpp
        interactor_scalar_swapped.hpp
        interactor_sse.hpp
        interactor_sse_swapped.hpp
        nbody_container.hpp
        nbody_initializer.hpp
        run_simulation.hpp
    COMPILE_FLAGS "-DNO_OMP -DNO_MPI -DHPX_LIMIT=15"
    DEPENDENCIES
        geodecomp
    FOLDER "nbody HPX Weak"
)

# add_hpx_executable(
#     nbody_mpi_strong
#     ESSENTIAL
#     SOURCES main_strong.cpp
#     HEADERS
#         interactor_scalar.hpp
#         interactor_scalar_swapped.hpp
#         interactor_sse.hpp
#         interactor_sse_swapped.hpp
#         nbody_container.hpp
#         nbody_initializer.hpp
#         run_simulation.hpp
#     COMPILE_FLAGS -fopenmp
#     LINK_FLAGS -fopenmp
#     DEPENDENCIES
#         geodecomp
#     FOLDER "nbody MPI Strong"
# )
#
# add_hpx_executable(
#     nbody_hpx_strong
#     ESSENTIAL
#     SOURCES main_strong.cpp
#     HEADERS
#         interactor_scalar.hpp
#         interactor_scalar_swapped.hpp
#         interactor_sse.hpp
#         interactor_sse_swapped.hpp
#         nbody_container.hpp
#         nbody_initializer.hpp
#         run_simulation.hpp
#     COMPILE_FLAGS "-DNO_OMP -DNO_MPI -DHPX_LIMIT=15"
#     DEPENDENCIES
#         geodecomp
#     FOLDER "nbody HPX Strong"
# )
