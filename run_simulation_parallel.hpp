
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/foreach.hpp>

#include <libgeodecomp/misc/statistics.h>
#include <libgeodecomp/io/tracingwriter.h>

#if defined(NO_MPI) && defined(NO_OMP)
struct NumUpdateGroups
{
    std::size_t operator()() const
    {
#ifdef HPX_NATIVE_MIC
        return boost::lexical_cast<std::size_t>(hpx::get_config_entry("nbody.micUpdateGroups", "1"));
#else
        return boost::lexical_cast<std::size_t>(hpx::get_config_entry("nbody.hostUpdateGroups", "1"));
#endif
    }
};
#endif

template<typename CELL>
void runSimulation(Coord<3> dim)
{
    int outputFrequency = 1;
    int maxSteps = 200;

    NBodyInitializer<CELL> *init = new NBodyInitializer<CELL>(dim, maxSteps);

#ifndef NO_MPI
    if(MPILayer().rank() == 0)
#endif
    {
        std::cout << "running simulation\n";
    }
#ifdef NO_MPI
    
    HpxSimulator::HpxSimulator<CELL, HiParSimulator::RecursiveBisectionPartition<3> > sim(
        init,
        NumUpdateGroups(),
        0,//MPILayer().rank() ? 0 : new TracingBalancer(new NoOpBalancer()),
        maxSteps,
        1);
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
        std::cout <<  size << " cores\n"
                  << "dim: " << dim << "\n";
    }
    sim.addWriter(
        new TracingWriter<CELL>(outputFrequency, init->maxSteps()));

    hpx::util::high_resolution_timer timer;
    sim.init();//initialWeights(dim.prod(), size));
    double initTime = timer.elapsed();
#ifndef NO_MPI
    if(MPILayer().rank() == 0)
#endif
    {
        std::cout << "initialization done in " << initTime << " seconds\n";
    }
    timer.restart();
#ifdef NO_MPI
    std::vector<Statistics> updateGroupTimes = sim.runTimed();
#else
    sim.run();
    MPILayer().barrier();
#endif
    double seconds = timer.elapsed();


    double flops =
        // time steps * grid size
        
        // interactions per container update
        27. * static_cast<double>(CELL::SIZE) * static_cast<double>(CELL::SIZE) *
        // FLOPs per interaction
        (3. + 6. + 1. + 6.);
    double dimProd = static_cast<double>(dim.x()) * static_cast<double>(dim.y()) * static_cast<double>(dim.z());
    double gflops =maxSteps * dimProd * (flops / (seconds * 1e9));
#ifndef NO_MPI
    SuperVector<Statistics> updateGroupTimes = sim.gatherStatistics();
    
    if(MPILayer().rank() != 0)
    {
        return;
    }
#endif
    typedef
        boost::accumulators::accumulator_set<
            double, 
            boost::accumulators::features<
                boost::accumulators::tag::min,
                boost::accumulators::tag::max,
                boost::accumulators::tag::count,
                boost::accumulators::tag::mean,
                boost::accumulators::stats<
                    boost::accumulators::tag::variance(boost::accumulators::immediate)
                >
            >
        > AccumulatorType;

    AccumulatorType accTotal;
    AccumulatorType accComputeInner;
    AccumulatorType accComputeGhost;
    AccumulatorType accPatchProviders;
    AccumulatorType accPatchAccepters;
    BOOST_FOREACH(const Statistics& stat, updateGroupTimes)
    {
        accTotal(stat.totalTime);
        accComputeInner(stat.computeTimeInner);
        accComputeGhost(stat.computeTimeGhost);
        accPatchProviders(stat.patchProvidersTime);
        accPatchAccepters(stat.patchAcceptersTime);
    }
    
    double timeMin  = (boost::accumulators::min)(accTotal);
    double timeMax  = (boost::accumulators::max)(accTotal);
    double timeMean =  boost::accumulators::mean(accTotal);
    double timeVar =  std::sqrt(boost::accumulators::variance(accTotal));
    
    double computeInnerMin  = (boost::accumulators::min)(accComputeInner);
    double computeInnerMax  = (boost::accumulators::max)(accComputeInner);
    double computeInnerMean =  boost::accumulators::mean(accComputeInner);
    double computeInnerVar =  std::sqrt(boost::accumulators::variance(accComputeInner));
    
    double computeGhostMin  = (boost::accumulators::min)(accComputeGhost);
    double computeGhostMax  = (boost::accumulators::max)(accComputeGhost);
    double computeGhostMean =  boost::accumulators::mean(accComputeGhost);
    double computeGhostVar =  std::sqrt(boost::accumulators::variance(accComputeGhost));
    
    double patchProvidersMin  = (boost::accumulators::min)(accPatchProviders);
    double patchProvidersMax  = (boost::accumulators::max)(accPatchProviders);
    double patchProvidersMean =  boost::accumulators::mean(accPatchProviders);
    double patchProvidersVar =  std::sqrt(boost::accumulators::variance(accPatchProviders));
    
    double patchAcceptersMin  = (boost::accumulators::min)(accPatchAccepters);
    double patchAcceptersMax  = (boost::accumulators::max)(accPatchAccepters);
    double patchAcceptersMean =  boost::accumulators::mean(accPatchAccepters);
    double patchAcceptersVar =  std::sqrt(boost::accumulators::variance(accPatchAccepters));
    // need to swap min and max here as the minimum time gives the maximum FLOPS
    double gflopsMax  = maxSteps * dim.prod() * (flops / (timeMin * 1e9));
    double gflopsMin  = maxSteps * dim.prod() * (flops / (timeMax * 1e9));
    double gflopsMean = maxSteps * dim.prod() * (flops / (timeMean * 1e9));
        
#ifdef NO_MPI
    std::cout << "HPX Simulation finished in " << seconds << " seconds\n"
#else
    std::cout << "MPI Simulation finished in " << seconds << " seconds\n"
#endif
              << " GFLOPS: " << gflops
              << " min: " << gflopsMin
              << " max: " << gflopsMax
              << " mean: " << gflopsMean << "\n\n"
              << "                      "
              << std::left << std::setw(15) << "min"
              << std::left << std::setw(15) << "max"
              << std::left << std::setw(15) << "mean"
              << std::left << std::setw(15) << "stddev"
              << "\n"
              << "Simulation Time       "
              << std::left << std::setw(15) << timeMin
              << std::left << std::setw(15) << timeMax
              << std::left << std::setw(15) << timeMean
              << std::left << std::setw(15) << timeVar
              << "\n"
              << "Compute Time Inner    "
              << std::left << std::setw(15) << computeInnerMin 
              << std::left << std::setw(15) << computeInnerMax 
              << std::left << std::setw(15) << computeInnerMean
              << std::left << std::setw(15) << computeInnerVar
              << "\n"
              << "Compute Time Ghost    "
              << std::left << std::setw(15) << computeGhostMin 
              << std::left << std::setw(15) << computeGhostMax 
              << std::left << std::setw(15) << computeGhostMean
              << std::left << std::setw(15) << computeGhostVar
              << "\n"
              << "PatchProviders Time   "
              << std::left << std::setw(15) << patchProvidersMin 
              << std::left << std::setw(15) << patchProvidersMax 
              << std::left << std::setw(15) << patchProvidersMean
              << std::left << std::setw(15) << patchProvidersVar
              << "\n"
              << "PatchAccepters Time   "
              << std::left << std::setw(15) << patchAcceptersMin 
              << std::left << std::setw(15) << patchAcceptersMax 
              << std::left << std::setw(15) << patchAcceptersMean
              << std::left << std::setw(15) << patchAcceptersVar
              << "\n"
        << "----------------------------------------------------------------------\n";
}
