
#ifndef INTERACTOR_AVX_HPP
#define INTERACTOR_AVX_HPP

template<int CONTAINER_SIZE>
class InteractorAVX
{
public:
    template<typename CONTAINER>
    void operator()(CONTAINER *target, const CONTAINER& oldSelf, const CONTAINER& neighbor)
    {
        __m256 forceOffset = _mm256_set1_ps(FORCE_OFFSET);
#ifndef NO_OMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < CONTAINER_SIZE; i += 8) {
            __m256 oldSelfPosX = _mm256_load_ps(oldSelf.posX + i);
            __m256 oldSelfPosY = _mm256_load_ps(oldSelf.posY + i);
            __m256 oldSelfPosZ = _mm256_load_ps(oldSelf.posZ + i);
            __m256 myVelX = _mm256_load_ps(oldSelf.velX + i);
            __m256 myVelY = _mm256_load_ps(oldSelf.velY + i);
            __m256 myVelZ = _mm256_load_ps(oldSelf.velZ + i);

            for (int j = 0; j < CONTAINER_SIZE; ++j) {
                __m256 neighborPosX = _mm256_set1_ps(neighbor.posX[j]);
                __m256 neighborPosY = _mm256_set1_ps(neighbor.posY[j]);
                __m256 neighborPosZ = _mm256_set1_ps(neighbor.posZ[j]);
                __m256 deltaX = _mm256_sub_ps(oldSelfPosX, neighborPosX);
                __m256 deltaY = _mm256_sub_ps(oldSelfPosY, neighborPosY);
                __m256 deltaZ = _mm256_sub_ps(oldSelfPosZ, neighborPosZ);
                __m256 dist2 = _mm256_add_ps(forceOffset,
                                          _mm256_mul_ps(deltaX, deltaX));
                dist2 = _mm256_add_ps(dist2,
                                   _mm256_mul_ps(deltaY, deltaY));
                dist2 = _mm256_add_ps(dist2,
                                   _mm256_mul_ps(deltaZ, deltaZ));
                __m256 force = _mm256_rsqrt_ps(dist2);
                myVelX = _mm256_add_ps(myVelX, _mm256_mul_ps(force, deltaX));
                myVelY = _mm256_add_ps(myVelY, _mm256_mul_ps(force, deltaY));
                myVelZ = _mm256_add_ps(myVelZ, _mm256_mul_ps(force, deltaZ));
            }

            _mm256_store_ps(target->velX + i, myVelX);
            _mm256_store_ps(target->velY + i, myVelY);
            _mm256_store_ps(target->velZ + i, myVelZ);
        }
    }

    void updateInner(int i)
    {
    }

    static const char * name()
    {
        return "AVX";
    }
};

#endif
