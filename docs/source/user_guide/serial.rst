==========
Run Serial
==========

Here's how to use waterEntropy on the command-line using the ``runWaterEntropy.py`` script.

.. code-block:: bash

    waterEntropy -t tests/input_files/amber/arginine_solution/system.prmtop \\
    -c tests/input_files/amber/arginine_solution/system.nc

Topology and trajectory files are available in the ``tests/input_files`` directory.

Here's how to use waterEntropy via the API:

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
    start, end, step = 0, 4, 2

    # Calculate the entropy
    Sorient_dict, covariances, vibrations, frame_solvent_indices, n_frames = \
        GetSolvent.get_interfacial_water_orient_entropy(
        u, start, end, step,
        temperature=298, # default simulated system temperate is set to 298 Kelvin, change accordingly
    )

    print(f"Number of frames analysed: {n_frames}")

    # Print Sorient
    OR.print_Sorient_dicts(Sorient_dict)

    # Print Svib
    VIB.print_Svib_data(vibrations, covariances)
