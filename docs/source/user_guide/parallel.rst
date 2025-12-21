============
Run Parallel
============

We have implemented a parallel computing strategy utilising the dask-distributed software package. Specifically we have implemented the LocalCluster for running parallel analyses with WaterEntropy. The parallel implementation runs the whole calculation by frames, so multiple frames of MD simulation are processed at the same time, and these are allocated 1 frame for every physical CPU core you have access to on your machine.

Here's how to use run the parallel waterEntropy from the command-line using the ``runWaterEntropy.py`` script.

.. code-block:: bash

    waterEntropy -t tests/input_files/amber/arginine_solution/system.prmtop -c tests/input_files/amber/arginine_solution/system.nc --parallel

Topology and trajectory files are available in the ``tests/input_files`` directory.

It is also possible to launch the parallel version via the API:

.. code-block:: Python

    # load modules
    from MDAnalysis import Universe
    import waterEntropy.recipes.interfacial_solvent as GetSolvent
    import waterEntropy.entropy.vibrations as VIB
    import waterEntropy.entropy.orientations as OR
    import waterEntropy.maths.trig as TRIG

    # set paths for topology and trajectory files
    topology_path = "tests/input_files/amber/arginine_solution/system.prmtop"
    trajectory_path = "tests/input_files/amber/arginine_solution/system.nc"

    # load topology and trajectory
    u = Universe(topology_path, trajectory_path)

    # set the frames to be analysed
    start, end, step = 0, 10, 1

    # Calculate the entropy
    Sorient_dict, covariances, vibrations, frame_solvent_indices, n_frames = GetSolvent.get_interfacial_water_orient_entropy(
        u, start, end, step,
        parallel=True, # set True for parallel calculation
        temperature=298, # default simulated system temperate is set to 298 Kelvin, change accordingly
    )

    print(f"Number of frames analysed: {n_frames}")

    # Print Sorient
    OR.print_Sorient_dicts(Sorient_dict)

    # Print Svib
    VIB.print_Svib_data(vibrations, covariances)
