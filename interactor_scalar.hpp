
#ifndef INTERACTOR_SCALAR_HPP
#define INTERACTOR_SCALAR_HPP

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

    static const char * name()
    {
        return "Scalar";
    }
};

#endif
