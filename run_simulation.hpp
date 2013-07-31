
#ifndef RUN_SIMULATION_HPP
#define RUN_SIMULATION_HPP

template<typename CELL>
void runSimulation()
{
    int outputFrequency = 100;
    int maxSteps = 10;
    Coord<3> dim(3, 3, 3);

    MPI::Aint displacements[] = {0};
    MPI::Datatype memberTypes[] = {MPI::CHAR};
    int lengths[] = { sizeof(CELL) };
    MPI::Datatype objType;
    objType =
        MPI::Datatype::Create_struct(1, lengths, displacements, memberTypes);
    objType.Commit();


    NBodyInitializer<CELL> *init = new NBodyInitializer<CELL>(dim, maxSteps);

    // HiParSimulator::HiParSimulator<CELL, HiParSimulator::RecursiveBisectionPartition<3> > sim(
    //     init,
    //     MPILayer().rank() ? 0 : new TracingBalancer(new NoOpBalancer()),
    //     maxSteps,
    //     1,
    //     objType);

    SerialSimulator<CELL> sim(init);

    /*
    if (MPILayer().rank() == 0) {
        sim.addWriter(
            new TracingWriter<CELL>(outputFrequency, init->maxSteps()));
    }
    */

    hpx::util::high_resolution_timer timer;
    sim.run();
    double seconds = timer.elapsed();

    unsigned long long flops =
        // time steps * grid size
        
        // interactions per container update
        27 * CELL::SIZE * CELL::SIZE *
        // FLOPs per interaction
        (3 + 6 + 1 + 6);
    double gflops =maxSteps * dim.prod() * (flops / (seconds * 1e9));
    std::cout << CELL::name() << " GFLOPS: " << gflops << "\n"
              << "----------------------------------------------------------------------\n";
}

#endif
