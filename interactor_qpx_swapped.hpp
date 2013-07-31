
#ifndef INTERACTOR_QPX_SWAPPED_HPP
#define INTERACTOR_QPX_SWAPPED_HPP

template<int CONTAINER_SIZE>
class InteractorQPXSwapped
{
public:
    template<typename CONTAINER>
    void operator()(CONTAINER *target, const CONTAINER& oldSelf, const CONTAINER& neighbor)
    {
        vector4double forceOffset = {FORCE_OFFSET, FORCE_OFFSET, FORCE_OFFSET, FORCE_OFFSET};

#pragma omp parallel for schedule(static)
        for (int i = 0; i < CONTAINER_SIZE; ++i) {
            vector4double oldSelfPosX = {oldSelf.posX[i], oldSelf.posX[i], oldSelf.posX[i], oldSelf.posX[i]};
            vector4double oldSelfPosY = {oldSelf.posY[i], oldSelf.posY[i], oldSelf.posY[i], oldSelf.posY[i]};
            vector4double oldSelfPosZ = {oldSelf.posZ[i], oldSelf.posZ[i], oldSelf.posZ[i], oldSelf.posZ[i]};

            vector4double myVelX = {oldSelf.velX[i], 0, 0, 0};
            vector4double myVelY = {oldSelf.velY[i], 0, 0, 0};
            vector4double myVelZ = {oldSelf.velZ[i], 0, 0, 0};

            for (long j = 0; j < CONTAINER_SIZE; j += 4) {
                vector4double neighborPosX = vec_ld(j, (double*)neighbor.posX);
                vector4double neighborPosY = vec_ld(j, (double*)neighbor.posY);
                vector4double neighborPosZ = vec_ld(j, (double*)neighbor.posZ);

                vector4double deltaX = vec_sub(oldSelfPosX, neighborPosX);
                vector4double deltaY = vec_sub(oldSelfPosY, neighborPosY);
                vector4double deltaZ = vec_sub(oldSelfPosZ, neighborPosZ);

                vector4double dist2 = vec_add(forceOffset,
                                              vec_mul(deltaX, deltaX));
                dist2 = vec_add(dist2,
                                vec_mul(deltaY, deltaY));
                dist2 = vec_add(dist2,
                                vec_mul(deltaZ, deltaZ));
                // vector4double force = vec_rsqrte(dist2);
                vector4double force = dist2;
                myVelX = vec_add(myVelX, vec_mul(force, deltaX));
                myVelY = vec_add(myVelY, vec_mul(force, deltaY));
                myVelZ = vec_add(myVelZ, vec_mul(force, deltaZ));
            }

            double buf[4];
            vec_st(myVelX, 0, buf);
            target->velX[i] = buf[0] + buf[1] + buf[2] + buf[3];
            vec_st(myVelY, 0, buf);
            target->velY[i] = buf[0] + buf[1] + buf[2] + buf[3];
            vec_st(myVelZ, 0, buf);
            target->velZ[i] = buf[0] + buf[1] + buf[2] + buf[3];
        }
    }
};

#endif
