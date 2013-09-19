project(nbody_demo)

cmake_minimum_required(VERSION 2.8)

if(NOT HPX_ROOT)
    if(NOT $ENV{HPX_ROOT} STREQUAL "")
        set(HPX_ROOT $ENV{HPX_ROOT})
   else()
       message(FATAL_ERROR "vandouken needs HPX. Please set HPX_ROOT.")
   endif()
endif()

if(HPX_ROOT)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${HPX_ROOT}/share/cmake-2.8/Modules)
endif()

################################################################################
# enable solution folders for MSVC
if(MSVC)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()
################################################################################

find_package(HPX REQUIRED)

if(NOT LIBGEODECOMP_INCLUDE_DIR)
    if(NOT $ENV{LIBGEODECOMP_INCLUDE_DIR} STREQUAL "")
        set(LIBGEODECOMP_INCLUDE_DIR $ENV{LIBGEODECOMP_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "vandouken needs LibGeoDecomp. Please set LIBGEODECOMP_INCLUDE_DIR.")
    endif()
endif()

if(NOT LIBGEODECOMP_LIB_DIR)
    if(NOT $ENV{LIBGEODECOMP_LIB_DIR} STREQUAL "")
        set(LIBGEODECOMP_LIB_DIR $ENV{LIBGEODECOMP_LIB_DIR})
    else()
        message(FATAL_ERROR "vandouken needs LibGeoDecomp. Please set LIBGEODECOMP_LIB_DIR.")
    endif()
endif()

if(WIN32)
    set(LOCAL_LIBGEODECOMP_LINK_LIB local_libgeodecomp_lib)
else()
    set(LOCAL_LIBGEODECOMP_LINK_LIB geodecomp)
endif()

include_directories(.)
include_directories(${HPX_INCLUDE_DIR})
include_directories(${LIBGEODECOMP_INCLUDE_DIR})

set(HPX_RPATH "${CMAKE_INSTALL_PREFIX}/lib/hpx:${CMAKE_BINARY_DIR}/lib/hpx:${LIBGEODECOMP_LIB_DIR}")
link_directories(${CMAKE_BINARY_DIR}/lib/hpx)
link_directories(${HPX_LIBRARY_DIR})

link_directories(${LIBGEODECOMP_LIB_DIR})

add_definitions(-DQT_NO_KEYWORDS)

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
    COMPILE_FLAGS -openmp
    LINK_FLAGS -openmp
    DEPENDENCIES
        ${LOCAL_LIBGEODECOMP_LINK_LIB}
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
    COMPILE_FLAGS -openmp
    LINK_FLAGS -openmp
    DEPENDENCIES
        ${LOCAL_LIBGEODECOMP_LINK_LIB}
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
        ${LOCAL_LIBGEODECOMP_LINK_LIB}
    FOLDER "nbody HPX Weak"
)

add_hpx_executable(
    nbody_mpi_strong
    ESSENTIAL
    SOURCES main_strong.cpp
    HEADERS
        interactor_scalar.hpp
        interactor_scalar_swapped.hpp
        interactor_sse.hpp
        interactor_sse_swapped.hpp
        nbody_container.hpp
        nbody_initializer.hpp
        run_simulation.hpp
    COMPILE_FLAGS -openmp
    LINK_FLAGS -openmp
    DEPENDENCIES
        ${LOCAL_LIBGEODECOMP_LINK_LIB}
    FOLDER "nbody MPI Strong"
)

add_hpx_executable(
    nbody_hpx_strong
    ESSENTIAL
    SOURCES main_strong.cpp
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
        ${LOCAL_LIBGEODECOMP_LINK_LIB}
    FOLDER "nbody HPX Strong"
)
