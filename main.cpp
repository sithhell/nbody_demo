// g++ -Ofast -march=native -I /home/gentryx/libgeodecomp/src/ -L /home/gentryx/libgeodecomp/build/Linux-x86_64/ -lgeodecomp main.cpp -fopenmp -o nbody_demo && echo go && echo go && time ./nbody_demo
// g++ -Ofast -march=native -I /home/gentryx/libgeodecomp/src/ -L /home/gentryx/libgeodecomp/build/Linux-x86_64/ -lgeodecomp main.cpp -fopenmp -o nbody_demo && echo go && echo go && time LD_LIBRARY_PATH=/home/gentryx/libgeodecomp/build/Linux-x86_64/ ./nbody_demo
#include <libgeodecomp.h>
#include <pmmintrin.h>

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


template<int CONTAINER_SIZE, typename FLOAT>
class InteractorScalar
{
public:
    template<typename CONTAINER>
    void operator()(CONTAINER *target, const CONTAINER& oldSelf, const CONTAINER& neighbor)
    {
        for (int i = 0; i < CONTAINER_SIZE; ++i) {
            for (int j = 0; j < CONTAINER_SIZE; ++j) {
                FLOAT deltaX = oldSelf.posX[i] - neighbor.posX[j];
                FLOAT deltaY = oldSelf.posY[i] - neighbor.posY[j];
                FLOAT deltaZ = oldSelf.posZ[i] - neighbor.posZ[j];
                FLOAT dist2 =
                    FORCE_OFFSET +
                    deltaX * deltaX +
                    deltaY * deltaY +
                    deltaZ * deltaZ;
                FLOAT force = 1 / sqrt(dist2);
                target->velX[i] += force * deltaX;
                target->velY[i] += force * deltaY;
                target->velZ[i] += force * deltaZ;
            }
        }
    }
};

template<int CONTAINER_SIZE, typename FLOAT>
class InteractorScalarSwapped
{
public:
    template<typename CONTAINER>
    void operator()(CONTAINER *target, const CONTAINER& oldSelf, const CONTAINER& neighbor)
    {
        for (int j = 0; j < CONTAINER_SIZE; ++j) {
            for (int i = 0; i < CONTAINER_SIZE; ++i) {
                FLOAT deltaX = oldSelf.posX[i] - neighbor.posX[j];
                FLOAT deltaY = oldSelf.posY[i] - neighbor.posY[j];
                FLOAT deltaZ = oldSelf.posZ[i] - neighbor.posZ[j];
                FLOAT dist2 =
                    FORCE_OFFSET +
                    deltaX * deltaX +
                    deltaY * deltaY +
                    deltaZ * deltaZ;
                FLOAT force = 1 / sqrt(dist2);
                target->velX[i] += force * deltaX;
                target->velY[i] += force * deltaY;
                target->velZ[i] += force * deltaZ;
            }
        }
    }
};

template<int CONTAINER_SIZE>
class InteractorSSE
{
public:
    template<typename CONTAINER>
    void operator()(CONTAINER *target, const CONTAINER& oldSelf, const CONTAINER& neighbor)
    {
        __m128 forceOffset = _mm_set1_ps(FORCE_OFFSET);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < CONTAINER_SIZE; i += 4) {
            __m128 oldSelfPosX = _mm_load_ps(oldSelf.posX + i);
            __m128 oldSelfPosY = _mm_load_ps(oldSelf.posY + i);
            __m128 oldSelfPosZ = _mm_load_ps(oldSelf.posZ + i);
            __m128 myVelX = _mm_load_ps(oldSelf.velX + i);
            __m128 myVelY = _mm_load_ps(oldSelf.velY + i);
            __m128 myVelZ = _mm_load_ps(oldSelf.velZ + i);

            for (int j = 0; j < CONTAINER_SIZE; ++j) {
                __m128 neighborPosX = _mm_set1_ps(neighbor.posX[j]);
                __m128 neighborPosY = _mm_set1_ps(neighbor.posY[j]);
                __m128 neighborPosZ = _mm_set1_ps(neighbor.posZ[j]);
                __m128 deltaX = oldSelfPosX - neighborPosX;
                __m128 deltaY = oldSelfPosY - neighborPosY;
                __m128 deltaZ = oldSelfPosZ - neighborPosZ;
                __m128 dist2 = _mm_add_ps(forceOffset,
                                          _mm_mul_ps(deltaX, deltaX));
                dist2 = _mm_add_ps(dist2,
                                   _mm_mul_ps(deltaY, deltaY));
                dist2 = _mm_add_ps(dist2,
                                   _mm_mul_ps(deltaZ, deltaZ));
                __m128 force = _mm_rsqrt_ps(dist2);
                myVelX = _mm_add_ps(myVelX, _mm_mul_ps(force, deltaX));
                myVelY = _mm_add_ps(myVelY, _mm_mul_ps(force, deltaY));
                myVelZ = _mm_add_ps(myVelZ, _mm_mul_ps(force, deltaZ));
            }

            _mm_store_ps(target->velX + i, myVelX);
            _mm_store_ps(target->velY + i, myVelY);
            _mm_store_ps(target->velZ + i, myVelZ);
        }
    }
};

template<int CONTAINER_SIZE>
class InteractorSSESwapped
{
public:
    template<typename CONTAINER>
    void operator()(CONTAINER *target, const CONTAINER& oldSelf, const CONTAINER& neighbor)
    {
        __m128 forceOffset = _mm_set1_ps(FORCE_OFFSET);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < CONTAINER_SIZE; ++i) {
            __m128 oldSelfPosX = _mm_set1_ps(oldSelf.posX[i]);
            __m128 oldSelfPosY = _mm_set1_ps(oldSelf.posY[i]);
            __m128 oldSelfPosZ = _mm_set1_ps(oldSelf.posZ[i]);
            __m128 myVelX = _mm_set_ps(oldSelf.velX[i], 0, 0, 0);
            __m128 myVelY = _mm_set_ps(oldSelf.velY[i], 0, 0, 0);
            __m128 myVelZ = _mm_set_ps(oldSelf.velZ[i], 0, 0, 0);

            for (int j = 0; j < CONTAINER_SIZE; j += 4) {
                __m128 neighborPosX = _mm_load_ps(neighbor.posX + j);
                __m128 neighborPosY = _mm_load_ps(neighbor.posY + j);
                __m128 neighborPosZ = _mm_load_ps(neighbor.posZ + j);
                __m128 deltaX = oldSelfPosX - neighborPosX;
                __m128 deltaY = oldSelfPosY - neighborPosY;
                __m128 deltaZ = oldSelfPosZ - neighborPosZ;
                __m128 dist2 = _mm_add_ps(forceOffset,
                                          _mm_mul_ps(deltaX, deltaX));
                dist2 = _mm_add_ps(dist2,
                                   _mm_mul_ps(deltaY, deltaY));
                dist2 = _mm_add_ps(dist2,
                                   _mm_mul_ps(deltaZ, deltaZ));
                __m128 force = _mm_rsqrt_ps(dist2);
                myVelX = _mm_add_ps(myVelX, _mm_mul_ps(force, deltaX));
                myVelY = _mm_add_ps(myVelY, _mm_mul_ps(force, deltaY));
                myVelZ = _mm_add_ps(myVelZ, _mm_mul_ps(force, deltaZ));
            }

            float buf[4];
            _mm_storeu_ps(buf, myVelX);
            target->velX[i] = buf[0] + buf[1] + buf[2] + buf[3];
            _mm_storeu_ps(buf, myVelY);
            target->velY[i] = buf[0] + buf[1] + buf[2] + buf[3];
            _mm_storeu_ps(buf, myVelZ);
            target->velZ[i] = buf[0] + buf[1] + buf[2] + buf[3];
        }
    }
};

template<int CONTAINER_SIZE, typename FLOAT_TYPE, typename INTERACTOR>
class NBodyContainer
{
public:
    typedef FLOAT_TYPE FLOAT;
    static const int SIZE = CONTAINER_SIZE;

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

#define INTERACT(REL_X, REL_Y, REL_Z) \
        INTERACTOR()(this, oldSelf, hood[FixedCoord<REL_X, REL_Y, REL_Z>()]);

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

    FLOAT posX[CONTAINER_SIZE];
    FLOAT posY[CONTAINER_SIZE];
    FLOAT posZ[CONTAINER_SIZE];
    FLOAT velX[CONTAINER_SIZE];
    FLOAT velY[CONTAINER_SIZE];
    FLOAT velZ[CONTAINER_SIZE];
};

template<typename CELL>
class NBodyInitializer : public SimpleInitializer<CELL>
{
public:
    using SimpleInitializer<CELL>::gridDimensions;

    NBodyInitializer(const Coord<3>& dim, unsigned steps) :
        SimpleInitializer<CELL>(dim, steps)
    {}

    virtual void grid(GridBase<CELL, 3> *ret)
    {
        CoordBox<3> box = ret->boundingBox();
        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            ret->at(*i) = CELL(*i);
        }
    }
};

template<typename CELL>
void runSimulation()
{
    int outputFrequency = 100;
    int maxSteps = 400;
    Coord<3> dim(3, 3, 3);

    MPI::Aint displacements[] = {0};
    MPI::Datatype memberTypes[] = {MPI::CHAR};
    int lengths[] = { sizeof(CELL) };
    MPI::Datatype objType;
    objType =
        MPI::Datatype::Create_struct(1, lengths, displacements, memberTypes);
    objType.Commit();


    NBodyInitializer<CELL> *init = new NBodyInitializer<CELL>(dim, maxSteps);

    HiParSimulator::HiParSimulator<CELL, HiParSimulator::RecursiveBisectionPartition<3> > sim(
        init,
        MPILayer().rank() ? 0 : new TracingBalancer(new NoOpBalancer()),
        maxSteps,
        1,
        objType);

    if (MPILayer().rank() == 0) {
        sim.addWriter(
            new TracingWriter<CELL>(outputFrequency, init->maxSteps()));
    }

    long long tStart = Chronometer::timeUSec();
    sim.run();
    long long tEnd = Chronometer::timeUSec();

    double seconds = 1e-6 * (tEnd - tStart);
    double flops =
        // time steps * grid size
        1.0 * maxSteps * dim.prod() *
        // interactions per container update
        27 * CELL::SIZE * CELL::SIZE *
        // FLOPs per interaction
        (3 + 6 + 1 + 6);
    double gflops = 1e-9 * flops / seconds;
    std::cout << "GFLOPS: " << gflops << "\n";
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    Typemaps::initializeMaps();

    runSimulation<NBodyContainer<512, float, InteractorSSE<512> > >();
    // runSimulation<NBodyContainer<512, float, InteractorSSESwapped<512> > >();

    MPI_Finalize();
    return 0;
}
