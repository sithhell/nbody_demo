
template<typename CELL>
void runSimulation(Coord<3> dim)
{
    int outputFrequency = 1;
    int maxSteps = 200;

    NBodyInitializer<CELL> *init = new NBodyInitializer<CELL>(dim, maxSteps);

#ifdef NO_MPI
    std::cout << "running simulation\n";
    HpxSimulator::HpxSimulator<CELL, HiParSimulator::RecursiveBisectionPartition<3> > sim(
        init,
        1, // overcommit Factor
        0,//MPILayer().rank() ? 0 : new TracingBalancer(new NoOpBalancer()),
        maxSteps,
        1);
    sim.initSimulation();
    std::cout << "initialization done\n";
    std::size_t size = hpx::get_num_worker_threads();
#else
    MPI::Aint displacements[] = {0};
    MPI::Datatype memberTypes[] = {MPI::CHAR};
    int lengths[] = { sizeof(CELL) };
    MPI::Datatype objType;
    objType =
        MPI::Datatype::Create_struct(1, lengths, displacements, memberTypes);
    objType.Commit();
    HiParSimulator::HiParSimulator<CELL, HiParSimulator::RecursiveBisectionPartition<3> > sim(
        init,
        0,//MPILayer().rank() ? 0 : new TracingBalancer(new NoOpBalancer()),
        maxSteps,
        1,
        objType);

    std::size_t size = MPILayer().size();

    if (MPILayer().rank() == 0)
#endif
    {
        std::cout << "ranks: " <<  size << "\n"
                  << "dim: " << dim << "\n";
        /*
        sim.addWriter(
            new TracingWriter<CELL>(outputFrequency, init->maxSteps()));
        */
    }

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
#ifndef NO_MPI
    if(MPILayer().rank() == 0)
#endif
        std::cout << CELL::name() << " GFLOPS: " << gflops << "\n"
                  << "----------------------------------------------------------------------\n";
}
