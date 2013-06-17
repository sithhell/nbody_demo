// g++ -O2 -march=native -I /home/gentryx/libgeodecomp/src/ -L /home/gentryx/libgeodecomp/build/Linux-x86_64/ -lgeodecomp main.cpp -fopenmp -o nbody_demo && echo go && time ./nbody_demo
#include <libgeodecomp.h>

using namespace LibGeoDecomp;

// fixme: tune
// - loop order
// - block size
// - manual vectorization vs. automatic


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

#define RECIPROCAL_SQUARE_ROOT(x) (1.0 / sqrt(x))
#define FLOAT float

typedef double FloatType;
const int CONTAINER_SIZE = 512;

class NBodyContainer
{
public:
    static const int FORCE_OFFSET = 0.01;
    typedef Stencils::Moore<3, 1> Stencil;
    typedef Topologies::Cube<3>::Topology Topology;
    class API : public CellAPITraits::Fixed
    {};

    static inline unsigned nanoSteps()
    {
        return 1;
    }

    inline NBodyContainer()
    {}

    inline NBodyContainer(const Coord<3>& pos)
    {
        for (int i; i < CONTAINER_SIZE; ++i) {
            posX[i] = (i         % 10) * 0.1 + pos.x();
            posY[i] = ((i / 10)  % 10) * 0.1 + pos.y();
            posZ[i] = ((i / 100) % 10) * 0.1 + pos.z();
            velX[i] = pos.x();
            velY[i] = pos.y();
            velZ[i] = pos.z();
        }
    }

    template<typename COORD_MAP>
    void update(const COORD_MAP& hood, const unsigned& nanoStep)
    {
        const NBodyContainer& oldSelf = hood[FixedCoord<0, 0, 0>()];

        for (int i = 0; i < CONTAINER_SIZE; ++i) {
            velX[i] = oldSelf.velX[i];
            velY[i] = oldSelf.velY[i];
            velZ[i] = oldSelf.velZ[i];
        }

#define INTERACT(REL_X, REL_Y, REL_Z)                                   \
        {                                                               \
            const NBodyContainer& neighbor =                            \
                hood[FixedCoord<REL_X, REL_Y, REL_Z>()];                \
            for (int i = 0; i < CONTAINER_SIZE; ++i) {                  \
                for (int j = 0; j < CONTAINER_SIZE; ++j) {              \
                    FLOAT deltaX = oldSelf.posX[i] - neighbor.posX[j];  \
                    FLOAT deltaY = oldSelf.posY[i] - neighbor.posY[j];  \
                    FLOAT deltaZ = oldSelf.posZ[i] - neighbor.posZ[j];  \
                    FLOAT dist2 =                                       \
                        FORCE_OFFSET +                                  \
                        deltaX * deltaX +                               \
                        deltaY * deltaY +                               \
                        deltaZ * deltaZ;                                \
                    FLOAT force = RECIPROCAL_SQUARE_ROOT(dist2);        \
                    velX[i] += force * deltaX;                          \
                    velY[i] += force * deltaY;                          \
                    velZ[i] += force * deltaZ;                          \
                }                                                       \
            }                                                           \
        }

        INTERACT(-1, -1, -1);
        INTERACT( 0, -1, -1);
        INTERACT( 1, -1, -1);
        INTERACT(-1,  0, -1);
        INTERACT( 0,  0, -1);
        INTERACT( 1,  0, -1);
        INTERACT(-1,  1, -1);
        INTERACT( 0,  1, -1);
        INTERACT( 1,  1, -1);

        INTERACT(-1, -1,  0);
        INTERACT( 0, -1,  0);
        INTERACT( 1, -1,  0);
        INTERACT(-1,  0,  0);
        INTERACT( 0,  0,  0);
        INTERACT( 1,  0,  0);
        INTERACT(-1,  1,  0);
        INTERACT( 0,  1,  0);
        INTERACT( 1,  1,  0);

        INTERACT(-1, -1,  1);
        INTERACT( 0, -1,  1);
        INTERACT( 1, -1,  1);
        INTERACT(-1,  0,  1);
        INTERACT( 0,  0,  1);
        INTERACT( 1,  0,  1);
        INTERACT(-1,  1,  1);
        INTERACT( 0,  1,  1);
        INTERACT( 1,  1,  1);

        for (int i = 0; i < CONTAINER_SIZE; ++i) {
            posX[i] = oldSelf.posX[i] + velX[i];
            posY[i] = oldSelf.posY[i] + velY[i];
            posZ[i] = oldSelf.posZ[i] + velZ[i];
        }

    }

private:
    FLOAT posX[CONTAINER_SIZE];
    FLOAT posY[CONTAINER_SIZE];
    FLOAT posZ[CONTAINER_SIZE];
    FLOAT velX[CONTAINER_SIZE];
    FLOAT velY[CONTAINER_SIZE];
    FLOAT velZ[CONTAINER_SIZE];
};

class NBodyInitializer : public SimpleInitializer<NBodyContainer>
{
public:
    using SimpleInitializer<NBodyContainer>::gridDimensions;

    NBodyInitializer(const Coord<3>& dim, unsigned steps) :
        SimpleInitializer<NBodyContainer>(dim, steps)
    {}

    virtual void grid(GridBase<NBodyContainer, 3> *ret)
    {
        CoordBox<3> box = ret->boundingBox();
        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            ret->at(*i) = NBodyContainer(*i);
        }
    }
};

void runSimulation()
{
    int outputFrequency = 100;
    int maxSteps = 10 ;
    Coord<3> dim(3, 3, 3);

    MPI::Aint displacements[] = {0};
    MPI::Datatype memberTypes[] = {MPI::CHAR};
    int lengths[] = {sizeof(NBodyContainer)};
    MPI::Datatype objType;
    objType =
        MPI::Datatype::Create_struct(1, lengths, displacements, memberTypes);
    objType.Commit();


    NBodyInitializer *init = new NBodyInitializer(dim, maxSteps);

    HiParSimulator::HiParSimulator<NBodyContainer, HiParSimulator::RecursiveBisectionPartition<3> > sim(
        init,
        MPILayer().rank() ? 0 : new TracingBalancer(new NoOpBalancer()),
        maxSteps,
        1,
        objType);

    if (MPILayer().rank() == 0) {
        sim.addWriter(
            new TracingWriter<NBodyContainer>(outputFrequency, init->maxSteps()));
    }

    long long tStart = Chronometer::timeUSec();
    sim.run();
    long long tEnd = Chronometer::timeUSec();

    double seconds = 1e-6 * (tEnd - tStart);
    double flops =
        // time steps * grid size
        1.0 * maxSteps * dim.prod() *
        // interactions per container update
        27 * CONTAINER_SIZE * CONTAINER_SIZE *
        // FLOPs per interaction
        (3 + 6 + 1 + 6);
    double gflops = 1e-9 * flops / seconds;
    std::cout << "GFLOPS: " << gflops << "\n";
}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    Typemaps::initializeMaps();

    runSimulation();

    MPI_Finalize();
    return 0;
}
