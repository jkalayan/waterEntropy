============
HPC Parallel
============

Running on HPC machines is slightly more complex that the simple dask distributed parallelism gained by using dask-distributed alone. The LocalCluster implementation gained by setting the "--parallel" flag often will not launch on HPC platforms because of how they are configured. So we have implemented a HPC friendly option that uses both dask-distributed and dask-jobqueue. Due to the complexity of how HPC machines operate, it is possible to submit these workloads in several different ways, and there are several options for manual configuration of parameters. The below table shows the HPC specific CLI flags, what they do and their default values.

================ ============================================================================ ============
Flag             Description                                                                  Default
================ ============================================================================ ============
--conda-exec     conda/mamba executable override, use if having issues.                       autodetect
--conda-env      Override for name of environment to load.                                    autodetect
--conda-path     Override for conda install.                                                  autodetect
--hpc            Turn on HPC parallelism.                                                     False
--hpc-account    Set HPC project/account to use for submissions.                              empty string
--hpc-constraint Constraints to apply to job allocation, such as hardware generation.         empty string
--hpc-cores      Override for number of cores on requested per node.                          autodetect
--hpc-memory     Override for memory request.                                                 autodetect
--hpc-nodes      How many nodes should the job run on.                                        1
--hpc-processes  Override How many dask processes per node. This should usually match cores.  autodetect
--hpc-queue      Override for queue name to submit to.                                        standard
--hpc-qos        Override for QoS.                                                            empty string
--hpc-walltime   Override for wall time job should request                                    24:00:00
--submit         Have WaterEntropy submit itself to a HPC cluster                             False
================ ============================================================================ ============

The easiest option for users to launch this version of WaterEntropy, is from the commandline using our advanced auto-detection features to lookup and detect various hardware features and submit WaterEntropy to the scheduler. This is likely to be the most compliant with HPC policies of the pure CLI ways to launch. To do this you simply run:

.. code-block:: bash

    waterEntropy --file-topology example_inputs/BTN_longer_sims/BTN_solvated_box.prmtop \\
    --file-coords example_inputs/BTN_longer_sims/BTN_5000frames.nc \\
    --start 0 --end 512 --step 1 --hpc --hpc-nodes 4 --hpc-account c01-bio \\
    --hpc-qos standard --submit

This will submit a master job to the scheduler system which will run the WaterEntropy master process, a separate dask cluster will then be orchestrated to run the work, so you will see a single master job plus the number of nodes requested as dask-workers in the HPC queue.

It is possible to run the same command, without the "--submit" and this will run the master python process on the HPC head node, with only the dask-worker cluster being sent to the scheduler. You should note, that this will cause resources to be blocked for other users on the head node, which may be seen as bad practice or against policies on some facilities:

.. code-block:: bash

    waterEntropy --file-topology example_inputs/BTN_longer_sims/BTN_solvated_box.prmtop \\
    --file-coords example_inputs/BTN_longer_sims/BTN_5000frames.nc \\
    --start 0 --end 128 --step 1 --hpc --hpc-account c01-bio --hpc-qos standard

If you want more control over how the WaterEntropy master process is submitted then you can submit your own script like this:

.. code-block:: bash

    #!/bin/bash --login

    #SBATCH --job-name=waterentropy-test
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=1
    #SBATCH --time=24:00:00
    #SBATCH --account=c01-bio
    #SBATCH --partition=standard
    #SBATCH --qos=standard

    eval "$(/mnt/lustre/a2fs-nvme/work/c01/c01/jtg2/miniforge3/bin/conda shell.bash hook)"
    eval "$(mamba shell hook --shell bash)"
    mamba activate waterentropy

    srun waterEntropy --file-topology example_inputs/BTN_longer_sims/BTN_solvated_box.prmtop \\
    --file-coords example_inputs/BTN_longer_sims/BTN_5000frames.nc --start 0 --end 512 \\
    --step 1 --hpc --hpc-nodes 4 --hpc-account c01-bio --hpc-qos standard


Topology and trajectory files are available in the ``tests/input_files`` directory.
