#ifndef NBODY_CONTAINER_HPP
#define NBODY_CONTAINER_HPP

#include <boost/serialization/is_bitwise_serializable.hpp>

template<int CONTAINER_SIZE, typename FLOAT_TYPE, typename INTERACTOR>
class NBodyContainer
{
public:
    typedef FLOAT_TYPE FLOAT;
    static const int SIZE = CONTAINER_SIZE;

    typedef Stencils::Moore<3, 1> Stencil;
    typedef Topologies::Cube<3>::Topology Topology;
#if 0
    class API : public CellAPITraits::Fixed
    {};
#else
    class API : public CellAPITraits::Base
    {};
#endif

    static inline unsigned nanoSteps()
    {
        return 1;
    }

    static const char * name()
    {
        return INTERACTOR::name();
    }

    inline NBodyContainer()
    {}

    inline NBodyContainer(const Coord<3>& pos)
    {
        for (int i = 0; i < CONTAINER_SIZE; ++i) {
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

#pragma simd
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

        INTERACTOR().move(this, oldSelf);
    }

    FLOAT posX[CONTAINER_SIZE];
    FLOAT posY[CONTAINER_SIZE];
    FLOAT posZ[CONTAINER_SIZE];
    FLOAT velX[CONTAINER_SIZE];
    FLOAT velY[CONTAINER_SIZE];
    FLOAT velZ[CONTAINER_SIZE];

    template <typename ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & posX;
        ar & posY;
        ar & posZ;
        ar & velX;
        ar & velY;
        ar & velZ;
    }
};

namespace boost { namespace serialization {

    template<int CONTAINER_SIZE, typename FLOAT_TYPE, typename INTERACTOR>
    struct is_bitwise_serializable<NBodyContainer<CONTAINER_SIZE, FLOAT_TYPE, INTERACTOR> >
        : boost::mpl::true_
    {};

}}

#endif
