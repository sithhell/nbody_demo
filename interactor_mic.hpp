
#ifndef INTERACTOR_MIC_HPP
#define INTERACTOR_MIC_HPP

template<int CONTAINER_SIZE, typename FLOAT_TYPE>
class InteractorMIC;

template<int CONTAINER_SIZE>
class InteractorMIC<CONTAINER_SIZE, float>
{
public:
    template<typename CONTAINER>
    void operator()(CONTAINER *target, const CONTAINER& oldSelf, const CONTAINER& neighbor)
    {
        const __m512 forceOffset = _mm512_set1_ps(FORCE_OFFSET);
#ifndef NO_OMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < CONTAINER_SIZE; i += 16) {
            __m512 oldSelfPosX = _mm512_load_ps(oldSelf.posX + i);
            __m512 oldSelfPosY = _mm512_load_ps(oldSelf.posY + i);
            __m512 oldSelfPosZ = _mm512_load_ps(oldSelf.posZ + i);
            __m512 myVelX = _mm512_load_ps(oldSelf.velX + i);
            __m512 myVelY = _mm512_load_ps(oldSelf.velY + i);
            __m512 myVelZ = _mm512_load_ps(oldSelf.velZ + i);

            for (int j = 0; j < CONTAINER_SIZE; j += 16) {
                const float * neighborPosXVec = neighbor.posX + j;
                const float * neighborPosYVec = neighbor.posY + j;
                const float * neighborPosZVec = neighbor.posZ + j;

                __m512 neighborPosX;
                __m512 neighborPosY;
                __m512 neighborPosZ;
                
                __m512 deltaX;
                __m512 deltaY;
                __m512 deltaZ;
                __m512 dist2;
                __m512 force;

#define POS(N) \
                neighborPosX = _mm512_extload_ps(neighborPosXVec + N, _MM_UPCONV_PS_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);\
                neighborPosY = _mm512_extload_ps(neighborPosYVec + N, _MM_UPCONV_PS_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);\
                neighborPosZ = _mm512_extload_ps(neighborPosZVec + N, _MM_UPCONV_PS_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);\
\
                deltaX = _mm512_sub_ps(oldSelfPosX, neighborPosX);\
                deltaY = _mm512_sub_ps(oldSelfPosY, neighborPosY);\
                deltaZ = _mm512_sub_ps(oldSelfPosZ, neighborPosZ);\
                dist2 = _mm512_fmadd_ps(forceOffset, deltaX, deltaX);\
                dist2 = _mm512_fmadd_ps(dist2, deltaY, deltaY);\
                dist2 = _mm512_fmadd_ps(dist2, deltaZ, deltaZ);\
                __m512 BOOST_PP_CAT(force, N) = _mm512_rsqrt23_ps(dist2);\
                /**/
#define POS2(N)\
                myVelX = _mm512_fmadd_ps(myVelX, BOOST_PP_CAT(force, N), deltaX);\
                myVelY = _mm512_fmadd_ps(myVelY, BOOST_PP_CAT(force, N), deltaY);\
                myVelZ = _mm512_fmadd_ps(myVelZ, BOOST_PP_CAT(force, N), deltaZ)\
                /**/
                POS(0); POS(1); POS(2); POS(3); POS(4); POS(5); POS(6); POS(7);
                POS(8); POS(9); POS(10); POS(11); POS(12); POS(13); POS(14); POS(15);
                POS2(0); POS2(1); POS2(2); POS2(3); POS2(4); POS2(5); POS2(6); POS2(7);
                POS2(8); POS2(9); POS2(10); POS2(11); POS2(12); POS2(13); POS2(14); POS2(15);
#undef POS
#undef POS2
            }

            _mm512_storenrngo_ps(target->velX + i, myVelX);
            _mm512_storenrngo_ps(target->velY + i, myVelY);
            _mm512_storenrngo_ps(target->velZ + i, myVelZ);
        }
    }
    
    template<typename CONTAINER>
    void move(CONTAINER *target, const CONTAINER& oldSelf)
    {
        for (int i = 0; i < CONTAINER_SIZE; i += 16) {
            __m512 posX = _mm512_load_ps(oldSelf.posX + i);
            __m512 posY = _mm512_load_ps(oldSelf.posY + i);
            __m512 posZ = _mm512_load_ps(oldSelf.posZ + i);
            
            __m512 velX = _mm512_load_ps(target->velX + i);
            __m512 velY = _mm512_load_ps(target->velY + i);
            __m512 velZ = _mm512_load_ps(target->velZ + i);

            posX = _mm512_add_ps(posX, velX);
            posY = _mm512_add_ps(posY, velY);
            posZ = _mm512_add_ps(posZ, velZ);

            _mm512_store_ps(target->posX + i, posX);
            _mm512_store_ps(target->posY + i, posY);
            _mm512_store_ps(target->posZ + i, posZ);
        }
    }

    static const char * name()
    {
        return "MIC";
    }
};

template<int CONTAINER_SIZE>
class InteractorMIC<CONTAINER_SIZE, double>
{
public:
    template<typename CONTAINER>
    void operator()(CONTAINER *target, const CONTAINER& oldSelf, const CONTAINER& neighbor)
    {
        const __m512d forceOffset = _mm512_set1_ps(FORCE_OFFSET);
        const __m512d one = _mm512_set1_pd(1.0);
#ifndef NO_OMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < CONTAINER_SIZE; i += 8) {
            __m512d oldSelfPosX = _mm512_load_pd(oldSelf.posX + i);
            __m512d oldSelfPosY = _mm512_load_pd(oldSelf.posY + i);
            __m512d oldSelfPosZ = _mm512_load_pd(oldSelf.posZ + i);
            __m512d myVelX = _mm512_load_pd(oldSelf.velX + i);
            __m512d myVelY = _mm512_load_pd(oldSelf.velY + i);
            __m512d myVelZ = _mm512_load_pd(oldSelf.velZ + i);

            for (int j = 0; j < CONTAINER_SIZE; ++j) {
                __m512d neighborPosX = _mm512_set1_pd(neighbor.posX[j]);
                __m512d neighborPosY = _mm512_set1_pd(neighbor.posY[j]);
                __m512d neighborPosZ = _mm512_set1_pd(neighbor.posZ[j]);
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
            }

            _mm512_store_pd(target->velX + i, myVelX);
            _mm512_store_pd(target->velY + i, myVelY);
            _mm512_store_pd(target->velZ + i, myVelZ);
        }
    }

    void updateInner(int i)
    {
    }

    static const char * name()
    {
        return "MIC";
    }
};

#endif
