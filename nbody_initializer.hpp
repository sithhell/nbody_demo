
#ifndef NBODY_INITIALIZER_HPP
#define NBODY_INITIALIZER_HPP

#include <boost/serialization/export.hpp>

template<typename CELL>
class NBodyInitializer : public SimpleInitializer<CELL>
{
public:
    using SimpleInitializer<CELL>::gridDimensions;
    NBodyInitializer() :
        SimpleInitializer<CELL>(Coord<3>(0,0,0), 0)
    {}

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

    template <typename ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & boost::serialization::base_object<SimpleInitializer<CELL> >(*this);
    }
};

#endif
