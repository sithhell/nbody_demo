
#ifndef NBODY_INITIALIZER_HPP
#define NBODY_INITIALIZER_HPP

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

#endif
