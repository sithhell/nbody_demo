
#ifndef INTERACTOR_SSE_SWAPPED_HPP
#define INTERACTOR_SSE_SWAPPED_HPP

template<int CONTAINER_SIZE>
class InteractorSSESwapped
{
public:
    template<typename CONTAINER>
    void operator()(CONTAINER *target, const CONTAINER& oldSelf, const CONTAINER& neighbor)
    {
        __m128 forceOffset = _mm_set1_ps(FORCE_OFFSET);
#ifndef NO_OMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j < CONTAINER_SIZE; ++j) {
            __m128 neighborPosX = _mm_set1_ps(neighbor.posX[j]);
            __m128 neighborPosY = _mm_set1_ps(neighbor.posY[j]);
            __m128 neighborPosZ = _mm_set1_ps(neighbor.posZ[j]);

            for (int i = 0; i < CONTAINER_SIZE; i+=4) {
                __m128 oldSelfPosX = _mm_load_ps(oldSelf.posX + i);
                __m128 oldSelfPosY = _mm_load_ps(oldSelf.posY + i);
                __m128 oldSelfPosZ = _mm_load_ps(oldSelf.posZ + i);
                __m128 myVelX = _mm_load_ps(oldSelf.velX + i);
                __m128 myVelY = _mm_load_ps(oldSelf.velY + i);
                __m128 myVelZ = _mm_load_ps(oldSelf.velZ + i);

                __m128 deltaX = _mm_sub_ps(oldSelfPosX, neighborPosX);
                __m128 deltaY = _mm_sub_ps(oldSelfPosY, neighborPosY);
                __m128 deltaZ = _mm_sub_ps(oldSelfPosZ, neighborPosZ);
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
                
                _mm_store_ps(target->velX + i, myVelX);
                _mm_store_ps(target->velY + i, myVelY);
                _mm_store_ps(target->velZ + i, myVelZ);
            }
        }
    }

    static const char * name()
    {
        return "SSE Swapped";
    }
};

#endif
