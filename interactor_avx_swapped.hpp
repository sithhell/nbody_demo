
#ifndef INTERACTOR_AVX_SWAPPED_HPP
#define INTERACTOR_AVX_SWAPPED_HPP

template<int CONTAINER_SIZE, typename FLOAT_TYPE>
class InteractorAVXSwapped;

template<int CONTAINER_SIZE>
class InteractorAVXSwapped<CONTAINER_SIZE, float>
{
public:
    template<typename CONTAINER>
    void operator()(CONTAINER *target, const CONTAINER& oldSelf, const CONTAINER& neighbor)
    {
        const __m256 forceOffset = _mm256_set1_ps(FORCE_OFFSET);

#ifndef NO_OMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j < CONTAINER_SIZE; ++j) {
            __m256 oldSelfPosX = _mm256_set1_ps(oldSelf.posX[j]);
            __m256 oldSelfPosY = _mm256_set1_ps(oldSelf.posY[j]);
            __m256 oldSelfPosZ = _mm256_set1_ps(oldSelf.posZ[j]);

            __m256 myVelX = _mm256_set1_ps(0);
            __m256 myVelY = _mm256_set1_ps(0);
            __m256 myVelZ = _mm256_set1_ps(0);

            for (int i = 0; i < CONTAINER_SIZE; i+=8) {
                __m256 neighborPosX = _mm256_load_ps(neighbor.posX + i);
                __m256 neighborPosY = _mm256_load_ps(neighbor.posY + i);
                __m256 neighborPosZ = _mm256_load_ps(neighbor.posZ + i);

                __m256 deltaX = _mm256_sub_ps(oldSelfPosX, neighborPosX);
                __m256 deltaY = _mm256_sub_ps(oldSelfPosY, neighborPosY);
                __m256 deltaZ = _mm256_sub_ps(oldSelfPosZ, neighborPosZ);
                __m256 dist2 = _mm256_add_ps(forceOffset, _mm256_mul_ps(deltaX, deltaX));
                dist2 = _mm256_add_ps(dist2, _mm256_mul_ps(deltaY, deltaY));
                dist2 = _mm256_add_ps(dist2, _mm256_mul_ps(deltaZ, deltaZ));
                __m256 force = _mm256_rsqrt_ps(dist2);
                myVelX = _mm256_add_ps(myVelX, _mm256_mul_ps(force, deltaX));
                myVelY = _mm256_add_ps(myVelY, _mm256_mul_ps(force, deltaY));
                myVelZ = _mm256_add_ps(myVelZ, _mm256_mul_ps(force, deltaZ));
            }

            target->velX[j] = oldSelf.velX[j];
            target->velY[j] = oldSelf.velY[j];
            target->velZ[j] = oldSelf.velZ[j];

            for (int index = 0; index < 8; ++index) {
                target->velX[j] += reinterpret_cast<float*>(&myVelX)[index];
                target->velY[j] += reinterpret_cast<float*>(&myVelY)[index];
                target->velZ[j] += reinterpret_cast<float*>(&myVelZ)[index];
            }

            target->posX[j] = oldSelf.posX[j] + target->velX[j];
            target->posY[j] = oldSelf.posY[j] + target->velY[j];
            target->posZ[j] = oldSelf.posZ[j] + target->velZ[j];
        }
    }

    template<typename CONTAINER>
    void move(CONTAINER *target, const CONTAINER& oldSelf)
    {
    }

    static const char * name()
    {
        return "AVX Swapped";
    }
};

template<int CONTAINER_SIZE>
class InteractorAVXSwapped<CONTAINER_SIZE, double>
{
public:
    template<typename CONTAINER>
    void operator()(CONTAINER *target, const CONTAINER& oldSelf, const CONTAINER& neighbor)
    {
        const __m256d forceOffset = _mm256_set1_pd(FORCE_OFFSET);
        const __m256d one = _mm256_set1_pd(1.0);
#ifndef NO_OMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j < CONTAINER_SIZE; ++j) {
            __m256d neighborPosX = _mm256_set1_pd(neighbor.posX[j]);
            __m256d neighborPosY = _mm256_set1_pd(neighbor.posY[j]);
            __m256d neighborPosZ = _mm256_set1_pd(neighbor.posZ[j]);

            for (int i = 0; i < CONTAINER_SIZE; i+=4) {
                __m256d oldSelfPosX = _mm256_load_pd(oldSelf.posX + i);
                __m256d oldSelfPosY = _mm256_load_pd(oldSelf.posY + i);
                __m256d oldSelfPosZ = _mm256_load_pd(oldSelf.posZ + i);
                __m256d myVelX = _mm256_load_pd(oldSelf.velX + i);
                __m256d myVelY = _mm256_load_pd(oldSelf.velY + i);
                __m256d myVelZ = _mm256_load_pd(oldSelf.velZ + i);

                __m256d deltaX = _mm256_sub_pd(oldSelfPosX, neighborPosX);
                __m256d deltaY = _mm256_sub_pd(oldSelfPosY, neighborPosY);
                __m256d deltaZ = _mm256_sub_pd(oldSelfPosZ, neighborPosZ);
                __m256d dist2 = _mm256_add_pd(forceOffset,
                                          _mm256_mul_pd(deltaX, deltaX));
                dist2 = _mm256_add_pd(dist2,
                                   _mm256_mul_pd(deltaY, deltaY));
                dist2 = _mm256_add_pd(dist2,
                                   _mm256_mul_pd(deltaZ, deltaZ));
                __m256d force = _mm256_div_pd(one, _mm256_sqrt_pd(dist2));
                myVelX = _mm256_add_pd(myVelX, _mm256_mul_pd(force, deltaX));
                myVelY = _mm256_add_pd(myVelY, _mm256_mul_pd(force, deltaY));
                myVelZ = _mm256_add_pd(myVelZ, _mm256_mul_pd(force, deltaZ));

                _mm256_store_pd(target->velX + i, myVelX);
                _mm256_store_pd(target->velY + i, myVelY);
                _mm256_store_pd(target->velZ + i, myVelZ);
            }

            target->velX[j] = oldSelf->velX[j];
            target->velY[j] = oldSelf->velY[j];
            target->velZ[j] = oldSelf->velZ[j];
            for (int index = 0; index < 8; ++index) {
                target->velX[j] += reinterpret_cast<float*>(&myVelX)[index];
                target->velY[j] += reinterpret_cast<float*>(&myVelY)[index];
                target->velZ[j] += reinterpret_cast<float*>(&myVelZ)[index];
            }

            target->posX[j] = oldSelf->posX[j] + target->velX[j];
            target->posY[j] = oldSelf->posY[j] + target->velY[j];
            target->posZ[j] = oldSelf->posZ[j] + target->velZ[j];
        }
    }

    static const char * name()
    {
        return "AVX Swapped";
    }
};

#endif
