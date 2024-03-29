# Cardio-PINN

**Cardio-PINN** is a framework to generate personalised functional left ventricular modes with physics informed
neural networks (PINN) as described in 

Buoso S, Joyce T and Kozerke S (2021) 'Personalising left-ventricular biophysical models of the heart using parametric physics-informed neural networks', Medical Image Analysis, [doi.org/10.1016/j.media.2021.102066](https://doi.org/10.1016/j.media.2021.102066). Please cite this work if you use this framework, or part of it such as the PINN setup, shape model, ...

The scripts in this repository allow to:

* Generate a synthetic left-ventricular anatomy and the corresponding parametrization using a shape model
* Train a PINN on a specific anatomy and run a simulation of left-ventricular function coupling the PINN to a 0D-circulation model. 

The repository contains:

* Shape model folder: it containts the bases defining the left-ventricular shape model, the range of the amplitudes for each basis in the training dataset and a vtk of the average anatomy with anatomical labels and physiological parametrization.
* Functional model folder: it contanins the bases defining the functional model and the ampitudes range for each basis obtained on a synthetic training dataset generated with a biophysical finite element model. 
* Generate_shapes.py: script for the generation of synthetic anatomies using the shape model
* CardioPINN.py: script for the training of a PINN on a selected left-ventricula anatomy and the simulation of the corresponding cardiac function when coupled to a 0D-circulation model. 
* DeepCardioFunctions.py: used defined functions for the CardioPINN.py script.

**Prerequisites**:
Scripts are tested with python version 3.5 and tensorflow version 1.10. To use the scripts you will need the following:

* Numpy
* Matplotlib
* Tensorflow
* Scipy
* Vtk

You can create the anaconda environment with the following sequence of commands

`conda create -n CardioPINN matplotlib scipy vtk ipython numpy`

`conda activate CardioPINN`

`conda install -c conda-forge tensorflow=1.10 `

# Synthetic shape generation:

Run `Generate_shapes.py` to generate a new synthetic shape using the shape model.
It is possible to specify directly the ampitudes of the bases or randomly
sample them between the limits of the values observed in the original dataset.

**Input data**:

    - POD_folder: path folder of the shape model 
    - n_modes: number of bases used
    - amp_vector: vector of the amplitudes of the bases, if amp_vector = None the
        code will randompy sample within the range defined in Amplitude_ranges.txt
    - output_folder: output path for storage of the anatomy

**Output vtk**:

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

# PINN training and simulation:

Run `CardioPINN.py` to train a PINN on a selected left-ventricular anatomy and simulate the corresponding cardiac function when coupled with a simplified systemic circulation model 

**Input data**:

    - cases_folder      : path to case folder (possibly the output folder of Generate_shapes.py)
    - case_name         : case name
    - endo_fiber_angle  : helix angle at endocardium [deg]
    - epi_fiber_angle   : helix angle at epicardium [deg]
    - gamma_angle       : orientation sheets [deg]
    - max_act           : maximum actuation stress value [Pa]
    - stiff_scale       : scaling value of shear moduli of the material model [-] 

    - Windkessel_R              : systemic circulation resistance
    - Windkessel_C              : systemic circulation compliance
    - end_diastolic_LV_pressure : end diastolic left ventricular pressure value
    - diastolic_aortic_pressure : end diastolic aortic pressure value
    
    - systole_length    : length of systole [ms]
    - diastole_length   : length of diastole [ms]
    - dt_               : time step [ms] (only to determine number of iterations)

    - n_input_variables     : number of input variables
    - n_modesU              : number of functional bases as last layer
    - hidden_layers         : number of hidden layers
    - hidden_neurons        : number of neurons per hidden layer
    - pressure_normalization:scaling value for pressure [mmHg]
    - stress_normalization  : scaling value for actuation stresses [Pa]

    - epochs        : number of training epocs
    - d_param       : number of points for tensor sampling of tuples (p_endo,T_a)  
    - learn_rate    : learning rate

# Usage example:
After setting all input parameters described, PINN can be trained on a synthetic anatomy generated using the shape model as:

`python Generate_shapes.py`

`python CardioPINN.py`

The code is developed by Dr. Stefano Buoso `buoso@biomed.ee.ethz.ch` [Cardiac Magnetic Resonance group](http://www.cmr.ethz.ch/), Institute for Biomedical Engineering, ETH Zurich, University of Zurich.
