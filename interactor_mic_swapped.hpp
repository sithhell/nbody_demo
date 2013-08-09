
#ifndef INTERACTOR_MIC_SWAPPED_HPP
#define INTERACTOR_MIC_SWAPPED_HPP

template<int CONTAINER_SIZE, typename FLOAT_TYPE>
class InteractorMICSwapped;

template<int CONTAINER_SIZE>
class InteractorMICSwapped<CONTAINER_SIZE, float>
{
public:
    template<typename CONTAINER>
    void operator()(CONTAINER *target, const CONTAINER& oldSelf, const CONTAINER& neighbor)
    {
        const __m512 forceOffset = _mm512_set1_ps(FORCE_OFFSET);

#ifndef NO_OMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j < CONTAINER_SIZE; ++j) {
            __m512 oldSelfPosX = _mm512_set1_ps(oldSelf.posX[j]);
            __m512 oldSelfPosY = _mm512_set1_ps(oldSelf.posY[j]);
            __m512 oldSelfPosZ = _mm512_set1_ps(oldSelf.posZ[j]);
            
            __m512 myVelX = _mm512_set1_ps(0);
            __m512 myVelY = _mm512_set1_ps(0);
            __m512 myVelZ = _mm512_set1_ps(0);
            
            for (int i = 0; i < CONTAINER_SIZE; i+=16) {
                __m512 neighborPosX = _mm512_load_ps(neighbor.posX + i);
                __m512 neighborPosY = _mm512_load_ps(neighbor.posY + i);
                __m512 neighborPosZ = _mm512_load_ps(neighbor.posZ + i);
                
                __m512 deltaX = _mm512_sub_ps(oldSelfPosX, neighborPosX);
                __m512 deltaY = _mm512_sub_ps(oldSelfPosY, neighborPosY);
                __m512 deltaZ = _mm512_sub_ps(oldSelfPosZ, neighborPosZ);
                __m512 dist2 = _mm512_fmadd_ps(forceOffset, deltaX, deltaX);
                dist2 = _mm512_fmadd_ps(dist2, deltaY, deltaY);
                dist2 = _mm512_fmadd_ps(dist2, deltaZ, deltaZ);
                __m512 force = _mm512_rsqrt23_ps(dist2);
                myVelX = _mm512_fmadd_ps(myVelX, force, deltaX);
                myVelY = _mm512_fmadd_ps(myVelY, force, deltaY);
                myVelZ = _mm512_fmadd_ps(myVelZ, force, deltaZ);
            }
            
            target->velX[j] = oldSelf.velX[j];
            target->velY[j] = oldSelf.velY[j];
            target->velZ[j] = oldSelf.velZ[j];
            for (int index = 0; index < 16; ++index) {
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
        return "MIC Swapped";
    }
};

template<int CONTAINER_SIZE>
class InteractorMICSwapped<CONTAINER_SIZE, double>
{
public:
    template<typename CONTAINER>
    void operator()(CONTAINER *target, const CONTAINER& oldSelf, const CONTAINER& neighbor)
    {
        const __m512d forceOffset = _mm512_set1_pd(FORCE_OFFSET);
        const __m512d one = _mm512_set1_pd(1.0);
#ifndef NO_OMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j < CONTAINER_SIZE; ++j) {
            __m512d neighborPosX = _mm512_set1_pd(neighbor.posX[j]);
            __m512d neighborPosY = _mm512_set1_pd(neighbor.posY[j]);
            __m512d neighborPosZ = _mm512_set1_pd(neighbor.posZ[j]);

            for (int i = 0; i < CONTAINER_SIZE; i+=8) {
                __m512d oldSelfPosX = _mm512_load_pd(oldSelf.posX + i);
                __m512d oldSelfPosY = _mm512_load_pd(oldSelf.posY + i);
                __m512d oldSelfPosZ = _mm512_load_pd(oldSelf.posZ + i);
                __m512d myVelX = _mm512_load_pd(oldSelf.velX + i);
                __m512d myVelY = _mm512_load_pd(oldSelf.velY + i);
                __m512d myVelZ = _mm512_load_pd(oldSelf.velZ + i);

                __m512d deltaX = _mm512_sub_pd(oldSelfPosX, neighborPosX);
                __m512d deltaY = _mm512_sub_pd(oldSelfPosY, neighborPosY);
                __m512d deltaZ = _mm512_sub_pd(oldSelfPosZ, neighborPosZ);
                __m512d dist2 = _mm512_add_pd(forceOffset,
                                          _mm512_mul_pd(deltaX, deltaX));
                dist2 = _mm512_add_pd(dist2,
                                   _mm512_mul_pd(deltaY, deltaY));
                dist2 = _mm512_add_pd(dist2,
                                   _mm512_mul_pd(deltaZ, deltaZ));
                __m512d force = _mm512_invsqrt_pd(dist2);
                myVelX = _mm512_add_pd(myVelX, _mm512_mul_pd(force, deltaX));
                myVelY = _mm512_add_pd(myVelY, _mm512_mul_pd(force, deltaY));
                myVelZ = _mm512_add_pd(myVelZ, _mm512_mul_pd(force, deltaZ));
                
                _mm512_store_pd(target->velX + i, myVelX);
                _mm512_store_pd(target->velY + i, myVelY);
                _mm512_store_pd(target->velZ + i, myVelZ);
            }
        }
    }

    static const char * name()
    {
        return "MIC Swapped";
    }
};

#endif
