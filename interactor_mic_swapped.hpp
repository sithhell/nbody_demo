
#ifndef INTERACTOR_MIC_SWAPPED_HPP
#define INTERACTOR_MIC_SWAPPED_HPP

template<int CONTAINER_SIZE>
class InteractorMICSwapped
{
public:
    template<typename CONTAINER>
    void operator()(CONTAINER *target, const CONTAINER& oldSelf, const CONTAINER& neighbor)
    {
        __m512 forceOffset = _mm512_set1_ps(FORCE_OFFSET);
#ifndef NO_OMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j < CONTAINER_SIZE; ++j) {
            __m512 neighborPosX = _mm512_set1_ps(neighbor.posX[j]);
            __m512 neighborPosY = _mm512_set1_ps(neighbor.posY[j]);
            __m512 neighborPosZ = _mm512_set1_ps(neighbor.posZ[j]);

            for (int i = 0; i < CONTAINER_SIZE; i+=16) {
                __m512 oldSelfPosX = _mm512_load_ps(oldSelf.posX + i);
                __m512 oldSelfPosY = _mm512_load_ps(oldSelf.posY + i);
                __m512 oldSelfPosZ = _mm512_load_ps(oldSelf.posZ + i);
                __m512 myVelX = _mm512_load_ps(oldSelf.velX + i);
                __m512 myVelY = _mm512_load_ps(oldSelf.velY + i);
                __m512 myVelZ = _mm512_load_ps(oldSelf.velZ + i);

                __m512 deltaX = _mm512_sub_ps(oldSelfPosX, neighborPosX);
                __m512 deltaY = _mm512_sub_ps(oldSelfPosY, neighborPosY);
                __m512 deltaZ = _mm512_sub_ps(oldSelfPosZ, neighborPosZ);
                __m512 dist2 = _mm512_add_ps(forceOffset,
                                          _mm512_mul_ps(deltaX, deltaX));
                dist2 = _mm512_add_ps(dist2,
                                   _mm512_mul_ps(deltaY, deltaY));
                dist2 = _mm512_add_ps(dist2,
                                   _mm512_mul_ps(deltaZ, deltaZ));
                //__m512 force = _mm512_rsqrt_ps(dist2);
                __m512 force = _mm512_rcp23_ps(dist2);
                myVelX = _mm512_add_ps(myVelX, _mm512_mul_ps(force, deltaX));
                myVelY = _mm512_add_ps(myVelY, _mm512_mul_ps(force, deltaY));
                myVelZ = _mm512_add_ps(myVelZ, _mm512_mul_ps(force, deltaZ));
                
                _mm512_store_ps(target->velX + i, myVelX);
                _mm512_store_ps(target->velY + i, myVelY);
                _mm512_store_ps(target->velZ + i, myVelZ);
            }
        }
    }

    static const char * name()
    {
        return "MIC Swapped";
    }
};

#endif
