/**

g++ -Ofast -march=native -I /home/gentryx/libgeodecomp/src/ -L /home/gentryx/libgeodecomp/build/Linux-x86_64/ -lgeodecomp main.cpp -fopenmp -o nbody_demo && echo go && echo go && time ./nbody_demo

g++ -Ofast -march=native -I /home/gentryx/libgeodecomp/src/ -L /home/gentryx/libgeodecomp/build/Linux-x86_64/ -lgeodecomp main.cpp -fopenmp -o nbody_demo && echo go && echo go && time LD_LIBRARY_PATH=/home/gentryx/libgeodecomp/build/Linux-x86_64/ ./nbody_demo

/bgsys/drivers/ppcfloor/comm/xl/bin/mpixlcxx   -I/bgsys/local/boost/1.47.0  -qthreaded -qhalt=e -O -DNDEBUG  main.cpp  -o main  -L/bgsys/local/boost/1.47.0/lib ~/libgeodecomp/build/Linux-ppc64/libgeodecomp.a -lboost_date_time-mt-1_47 -lboost_filesystem-mt-1_47 -lboost_system-mt-1_47 -lboost_thread-mt-1_47 -Wl,-Bstatic -lcxxmpich -lmpich -lopa -lmpl -Wl,-Bdynamic -lpami -Wl,-Bstatic -lSPI -lSPI_cnk -Wl,-Bdynamic -lrt -lpthread -lstdc++ -lpthread -lstdc++ -Wl,-rpath,/bgsys/local/boost/1.47.0/lib -I ~/libgeodecomp/src/ -qmaxmem=-1

*/
#include <mpi.h>
#include <hpx/config.hpp>
#if defined(NO_OMP) && defined(NO_MPI)
#include <libgeodecomp/parallelization/hpxsimulator.h>
#include <hpx/hpx_main.hpp>
#else
#include <omp.h>
#endif
#include <libgeodecomp.h>
#include <immintrin.h>

#include <hpx/util/high_resolution_timer.hpp>

using namespace LibGeoDecomp;

// fixme: tune
// - loop order
// - block size
// - manual vectorization vs. automatic
// - how many threads? 16 vs 32 vs 64

// void update() {
//     // fixme: make precision configurable
//     int threadID = threadIdx.x;
//     double posX = positionsX[threadID];
//     double posY = positionsY[threadID];
//     double posZ = positionsZ[threadID];

//     double forceX = 0.0;
//     double forceY = 0.0;
//     double forceZ = 0.0;

//     #pragma unroll
//     for (int i = 0; i < NUM_PARTICLES; ++i) {
//         double deltaX = positionsX[i] - posX;
//         double deltaY = positionsY[i] - posY;
//         double deltaZ = positionsZ[i] - posZ;
//         double deltaSquared = deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ + deltaOffset;
//         double deltaReciprocal = rdqrtf(deltaSquared);
//         double deltaReciprocal3 = deltaReciprocal * deltaReciprocal * deltaReciprocal;

//         forceX += deltaX * deltaReciprocal;
//         forceY += deltaY * deltaReciprocal;
//         forceZ += deltaZ * deltaReciprocal;
//     }

//     // fixme: read vel
//     velX += forceX * deltaT;
//     velY += forceY * deltaT;
//     velZ += forceZ * deltaT;
//     posX += velX * deltaT;
//     posY += velY * deltaT;
//     posZ += velZ * deltaT;
//     // fixme: write back
// }

#define FORCE_OFFSET 0.01


#include "interactor_scalar.hpp"
#include "interactor_scalar_swapped.hpp"

#ifdef HPX_NATIVE_MIC
#include "interactor_mic.hpp"
#include "interactor_mic_swapped.hpp"
#else
#include "interactor_sse.hpp"
#include "interactor_sse_swapped.hpp"

#include "interactor_avx.hpp"
#include "interactor_avx_swapped.hpp"
#endif

//#include "interactor_qpx_swapped.hpp"

#include "nbody_container.hpp"
#include "nbody_initializer.hpp"

#if defined(NO_OMP) && defined(NO_MPI)
#ifdef HPX_NATIVE_MIC
typedef NBodyContainer<512, float, InteractorMIC<512, float> > CellType;
#else
typedef NBodyContainer<512, float, InteractorAVX<512, float> > CellType;
#endif
typedef NBodyInitializer<CellType> NBodyInitializerType;
typedef HpxSimulator::HpxSimulator<CellType, HiParSimulator::RecursiveBisectionPartition<3> > SimulatorType;
BOOST_CLASS_EXPORT_GUID(NBodyInitializerType, "NBodyInitializer");

LIBGEODECOMP_REGISTER_HPX_SIMULATOR_DECLARATION(
    SimulatorType
  , NBodySimulator
);

LIBGEODECOMP_REGISTER_HPX_SIMULATOR(
    SimulatorType
  , NBodySimulator
);
#endif


#include "run_simulation_parallel.hpp"


int main(int argc, char **argv)
{
#ifndef NO_MPI
    MPI_Init(&argc, &argv);
    Typemaps::initializeMaps();
    std::size_t size = MPILayer().size();
#else
    std::size_t size = hpx::get_num_worker_threads();
#endif
    
    Coord<3> dim(60, 60, 60);

#ifdef HPX_NATIVE_MIC
        runSimulation<NBodyContainer<512, float,  InteractorMIC<512, float> > >(dim);
#else
        runSimulation<NBodyContainer<512, float,  InteractorAVX<512, float> > >(dim);
#endif
        // runSimulation<NBodyContainer<512, float, InteractorQPXSwapped<512> > >(dim);

/*
#ifdef HPX_NATIVE_MIC
        runSimulation<NBodyContainer<512, float,  InteractorMIC<512> > >(dim);
#else
        runSimulation<NBodyContainer<512, float,  InteractorAVX<512> > >(dim);
#endif
        // runSimulation<NBodyContainer<512, float, InteractorQPXSwapped<512> > >(dim);

#ifdef HPX_NATIVE_MIC
        runSimulation<NBodyContainer<512, float,  InteractorMIC<512> > >(dim);
#else
        runSimulation<NBodyContainer<512, float,  InteractorAVX<512> > >(dim);
#endif
        // runSimulation<NBodyContainer<512, float, InteractorQPXSwapped<512> > >(dim);

#ifdef HPX_NATIVE_MIC
        runSimulation<NBodyContainer<1024, float,  InteractorMIC<1024> > >(dim);
#else
        runSimulation<NBodyContainer<1024, float,  InteractorAVX<1024> > >(dim);
#endif
        // runSimulation<NBodyContainer<1024, float, InteractorQPXSwapped<1024> > >(dim);
*/

#ifndef NO_MPI
    MPI_Finalize();
#endif
    return 0;
}
