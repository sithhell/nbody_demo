#ifndef NBODY_CONTAINER_HPP
#define NBODY_CONTAINER_HPP

#include <boost/serialization/is_bitwise_serializable.hpp>

template<int CONTAINER_SIZE, typename FLOAT_TYPE, typename INTERACTOR>
class __attribute__((aligned(64)))  NBodyContainer
{
public:
    typedef FLOAT_TYPE FLOAT;
    static const int SIZE = CONTAINER_SIZE;
    
    class API
        : public APITraits::HasFixedCoordsOnlyUpdate
        //, public APITraits::HasSpeed
        , public APITraits::HasStencil<Stencils::Moore<3, 1> >
        , public APITraits::HasCubeTopology<3>
    {};

    static const char * name()
    {
        return INTERACTOR::name();
    }

    static double speed()
    {
        char * speedStr = std::getenv("SPEED");
        if(speedStr)
        {
            return boost::lexical_cast<double>(speedStr);
        }
        return 1.0;
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

//#pragma simd
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

    FLOAT __attribute__((aligned(64))) posX[CONTAINER_SIZE];
    FLOAT __attribute__((aligned(64))) posY[CONTAINER_SIZE];
    FLOAT __attribute__((aligned(64))) posZ[CONTAINER_SIZE];
    FLOAT __attribute__((aligned(64))) velX[CONTAINER_SIZE];
    FLOAT __attribute__((aligned(64))) velY[CONTAINER_SIZE];
    FLOAT __attribute__((aligned(64))) velZ[CONTAINER_SIZE];

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
