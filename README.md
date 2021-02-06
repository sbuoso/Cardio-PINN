# Cardio-PINN

**Cardio-PINN** is a framework to generate personalised functional left ventricular modes with physics informed
neural networks (PINN) as described in 

Buoso S, Joyce T and Kozerke S (2021) 'Personalising left-ventricular biophysical models of the heart using parametric physics-informed neural networks', Medical Image Analysis, xx, xx, xx,. Please cite this work if you use this framework, or part of it such as the PINN setup, shape model, ...

The scripts in this repository allow to:
* Generate a synthetic left-ventricular anatomy and the corresponding parametrization using a shape model
* Train a PINN on a specific anatomy and run a simulation of left-ventricular function coupling the PINN to a 0D-circulation model. 

**Synthetic shape generation**:
Synthetic shapes can be generated running the script Generate_shapes.py
The shape model is consists of bases obtained applying Proper Orthogonal Decomposition
(POD) to a dataset of anatomies. New shapes can be generated as a linear combination
of the bases. It is possible to specify directly the ampitudes of the bases or randomly
sample them between the limits of the values observed in the original dataset.

Input data:
    - POD_folder: path folder of the shape model 
    - n_modes: number of bases used
    - amp_vector: vector of the amplitudes of the bases, if amp_vector = None the
        code will randompy sample within the range defined in Amplitude_ranges.txt
    - output_folder: output path for storage of the anatomy

The output is a vtk file in the form of an unstructured grid. The vtk defines:
    - coordinates of the points obtained with the shape model
    - labels of the points:
        . 1 = Endocardium
        . 2 = Epicardium
        . 3 = Mitral valve endocardium ring
        . 4 = Mitral valve internal
        . 5 = Mitral valve epicardium ring
        
    - parametric coordinates of the anatomy:
        . x_l = longitudinal coordinate parameter
        . x_c = circumferential coordinate parameter
        . x_t = transmural coordinate parameter

    - local physiological directions:
        . e_l = local longitudinal direction vector
        . e_c = local circumferential direction vector
        . e_t = local transmural direction vector


**Set up of the simulation environment**:
Scripts are tested with python version 3.5 and tensorflow version 1.10

You can create the anaconda environment with the following sequence of commands

`conda create -n CardioPINN matplotlib scipy vtk ipython numpy`
`conda activate CardioPINN`
`conda install -c conda-forge tensorflow=1.10 `

The code is developed by Dr. Stefano Buoso `buoso@biomed.ee.ethz.ch` [Cardiac Magnetic Resonance group](http://www.cmr.ethz.ch/), Institute for Biomedical Engineering, ETH Zurich, University of Zurich.
