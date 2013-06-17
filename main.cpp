// g++ -O2 -march=native -I /home/gentryx/libgeodecomp/src/ -L /home/gentryx/libgeodecomp/build/Linux-x86_64/ -lgeodecomp main.cpp -fopenmp -o sph_demo && echo go && time ./sph_demo
#include <libgeodecomp.h>

using namespace LibGeoDecomp;

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

class NBodyContainer
{
public:
    static const int SIZE = 256;

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

    inline NBodyContainer(const Coord<3>& pos, const Coord<3>& dim)
    {
        for (int i; i < SIZE; ++i) {
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
        const NBodyContainer& self = hood[FixedCoord<0, 0, 0>()];

        for (int i = 0; i < SIZE; ++i) {
            velX[i] = original.velX[i];
            velY[i] = original.velY[i];
            velZ[i] = original.velZ[i];
        }

#define INTERACT(REL_X, REL_Y, REL_Z)                                   \
        {                                                               \
            const NBodyContainer& neighbor =                            \
                hood[FixedCoord<REL_X, REL_Y, REL_Z>()];                \
            for (int i = 0; i < SIZE; ++i) {                            \
                for (int j = 0; j < SIZE; ++j) {                        \
                    double deltaX = self.posX[i] - neighbor.posX[j];    \
                    double deltaY = self.posY[i] - neighbor.posY[j];    \
                    double deltaZ = self.posZ[i] - neighbor.posZ[j];    \
                    double dist2 =                                      \
                        deltaX * deltaX +                               \
                        deltaY * deltaY +                               \
                        deltaZ * deltaZ;                                \
                    double force = RECIPROCAL_SQUARE_ROOT(dist2);       \
                    velX[i] += forse * deltaX;                          \
                    velY[i] += forse * deltaY;                          \
                    velZ[i] += forse * deltaZ;                          \
                }                                                       \
            }                                                           \
        }


        for (int i = 0; i < SIZE; ++i) {
            posX[i] = original.posX[i] + velX[i];
            posY[i] = original.posY[i] + velY[i];
            posZ[i] = original.posZ[i] + velZ[i];
        }

    }

private:
    double posX[SIZE];
    double posY[SIZE];
    double posZ[SIZE];
    double velX[SIZE];
    double velY[SIZE];
    double velZ[SIZE];
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
    int outputFrequency = 10;
    int maxSteps = 100;
    Coord<3> dim(3, 3, 3);

    NBodyInitializer *init = new NBodyInitializer(dim, maxSteps);

    HiParSimulator::HiParSimulator<NBodyContainer, HiParSimulator::RecursiveBisectionPartition<3> > sim(
        init,
        MPILayer().rank() ? 0 : new TracingBalancer(new NoOpBalancer()),
        maxSteps,
        1,
        MPI::DOUBLE);

    if (MPILayer().rank() == 0) {
        sim.addWriter(
            new TracingWriter<NBodyContainer>(outputFrequency, init->maxSteps()));
    }

    sim.run();
}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    Typemaps::initializeMaps();

    runSimulation();

    MPI_Finalize();
    return 0;
}
