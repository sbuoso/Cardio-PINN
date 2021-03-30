'''
Generation of synthetic anatomies using a shape model.
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

Copiright:  Buoso Stefano 2021. ETH Zurich
            buoso@biomed.ee.ethz.ch
'''

# Import packages
import vtk
import numpy as np
import os, sys
from   vtk.util.numpy_support import vtk_to_numpy

# Input section
local_path    = os.getcwd()
POD_folder    = local_path + '/Shape_model/POD_bases/'
out_path      = local_path + '/Synthetic_shapes/'
out_case      = '/Shape1/'

n_modes     = 4
ampl_vector = None

POD_files  = np.sort(next(os.walk(POD_folder))[2])
LV_mesh    = local_path + '/Shape_model/LV_mean.vtk'
ampl_file  = local_path + '/Shape_model/Amplitude_ranges.txt'
ampl_vector       = None

if ampl_vector is not None:
    assert len(ampl_vector) == n_modes

if not os.path.exists(out_path):
    os.makedirs(out_path)
if not os.path.exists(out_path+out_case):
    os.makedirs(out_path+out_case)

# Read bases for construction of mesh and directions
PHI = []
PHI_c = []
PHI_t = []
PHI_l = []
for m_sel in range(n_modes):
    Phi_matrix  = np.load(POD_folder+'/Phi'+str(m_sel)+'_points.npy')

#    if m_sel == 0:
#        PHI = np.concatenate((Phi_matrix[:,0].reshape(-1,1),Phi_matrix[:,1].reshape(-1,1),Phi_matrix[:,2].reshape(-1,1)),0)
#    else:
#        pp_sel = np.concatenate((Phi_matrix[:,0].reshape(-1,1),Phi_matrix[:,1].reshape(-1,1),Phi_matrix[:,2].reshape(-1,1)),0)
#        PHI    = np.concatenate((PHI,pp_sel),1)
    PHI.append(Phi_matrix)

    Phi_matrix  = np.load(POD_folder+'/Phi'+str(m_sel)+'_ec.npy')
    
    # if m_sel == 0:
    #     PHI_c = np.concatenate((Phi_matrix[:,0].reshape(-1,1),Phi_matrix[:,1].reshape(-1,1),Phi_matrix[:,2].reshape(-1,1)),0)
    # else:
    #     pp_sel = np.concatenate((Phi_matrix[:,0].reshape(-1,1),Phi_matrix[:,1].reshape(-1,1),Phi_matrix[:,2].reshape(-1,1)),0)
    #     PHI_c    = np.concatenate((PHI_c,pp_sel),1)
    PHI_c.append(Phi_matrix)

    Phi_matrix  = np.load(POD_folder+'/Phi'+str(m_sel)+'_el.npy')
    
    # if m_sel == 0:
    #     PHI_l = np.concatenate((Phi_matrix[:,0].reshape(-1,1),Phi_matrix[:,1].reshape(-1,1),Phi_matrix[:,2].reshape(-1,1)),0)
    # else:
    #     pp_sel = np.concatenate((Phi_matrix[:,0].reshape(-1,1),Phi_matrix[:,1].reshape(-1,1),Phi_matrix[:,2].reshape(-1,1)),0)
    #     PHI_l    = np.concatenate((PHI_l,pp_sel),1)
    PHI_l.append(Phi_matrix)

    Phi_matrix  = np.load(POD_folder+'/Phi'+str(m_sel)+'_et.npy')
    
    # if m_sel == 0:
    #     PHI_t = np.concatenate((Phi_matrix[:,0].reshape(-1,1),Phi_matrix[:,1].reshape(-1,1),Phi_matrix[:,2].reshape(-1,1)),0)
    # else:
    #     pp_sel = np.concatenate((Phi_matrix[:,0].reshape(-1,1),Phi_matrix[:,1].reshape(-1,1),Phi_matrix[:,2].reshape(-1,1)),0)
    #     PHI_t    = np.concatenate((PHI_t,pp_sel),1)
    PHI_t.append(Phi_matrix)

# Read vtk file with mean shape
source_mesh_vtk = vtk.vtkUnstructuredGridReader()
source_mesh_vtk.SetFileName(LV_mesh)
source_mesh_vtk.ReadAllScalarsOn()
source_mesh_vtk.Update()
source_mesh  = source_mesh_vtk.GetOutput()
n_points  = int(source_mesh.GetNumberOfPoints())
Coords_0    = vtk_to_numpy(source_mesh.GetPoints().GetData()).copy()
# Read parametric coordinates and labels
xc       = vtk_to_numpy(source_mesh.GetPointData().GetArray('x_c'))
xl       = vtk_to_numpy(source_mesh.GetPointData().GetArray('x_l'))
xt       = vtk_to_numpy(source_mesh.GetPointData().GetArray('x_t'))
label    = vtk_to_numpy(source_mesh.GetPointData().GetArray('labels'))
n_cells   = source_mesh.GetNumberOfCells()
Els = [] # line elements
# Get elements from VTK
for ii in range(n_cells):
    n_nodes_el   = source_mesh.GetCell(ii).GetPointIds().GetNumberOfIds()
    connectivity = [0]*n_nodes_el
    for n_sel in range(n_nodes_el):
        connectivity[n_sel] = int(source_mesh.GetCell(ii).GetPointId(n_sel)) 
    Els.append(connectivity)

Els = np.array(Els)

# Read ranges of amplitudes 
if ampl_vector == None:
    ampl_vector = [0]*n_modes
    ww = np.random.rand(n_modes)
    with open(ampl_file) as infile:
        coeff_files = infile.readlines()
    coeffs_boundaries = np.zeros((n_modes,2))
    for ii in range(n_modes):
        dd = coeff_files[ii].lstrip('Mode '+str(ii)+' coeffs: ').rstrip('\n').split(' - ')
        coeffs_boundaries[ii,0] = float(dd[0]) # min value
        coeffs_boundaries[ii,1] = float(dd[1]) # max value

        ampl_vector[ii] = coeffs_boundaries[ii,0]*ww[ii]+ coeffs_boundaries[ii,1]*(1- ww[ii])


Coords = np.zeros((n_points,3))
e_t = np.zeros((n_points,3))
e_l = np.zeros((n_points,3))
e_c = np.zeros((n_points,3))

for ii in range(n_modes):
    Coords += PHI[ii]*ampl_vector[ii]
    e_c += PHI_c[ii]*ampl_vector[ii]
    e_l += PHI_l[ii]*ampl_vector[ii]
    e_t += PHI_t[ii]*ampl_vector[ii]

outFile = open(out_path+out_case+'/Anatomy.vtk' , 'w' )
# write vtk header
outFile.write( '# vtk DataFile Version 4.0' )
outFile.write( '\nvtk output' )
outFile.write( '\nASCII' )
outFile.write( '\nDATASET UNSTRUCTURED_GRID' )  
outFile.write( '\n\nPOINTS ' + str( n_points ) + ' float\n' )
    
for i in range( n_points ):
    for j in range( 3 ):
        outFile.write( str( Coords[i,j] ) + ' ' )
    outFile.write( '\n' )

outFile.write( 'CELLS ' + str( n_cells ) + ' ' + str( n_cells * 5 ) )

for i in range( n_cells ):
    sel_element =  Els[i,:]
    outFile.write( '\n' )
    outFile.write(str(len(sel_element))+' ')
    for j in range( len(sel_element) ):
        outFile.write( str( sel_element[j]) + ' ' )
    
outFile.write( '\n\nCELL_TYPES ' + str(n_cells) )

for i in range( n_cells):
    outFile.write( '\n10' )

outFile.write( '\nPOINT_DATA ' + str( n_points )+'\n')
outFile.write( 'SCALARS labels float \n') # transmural position
outFile.write( 'LOOKUP_TABLE labels \n')
for i in range( n_points ):
    outFile.write(str(label[i]) )
    outFile.write( '\n' )
outFile.write( 'SCALARS x_t float \n') # transmural position
outFile.write( 'LOOKUP_TABLE x_t \n')
for i in range( n_points ):
    outFile.write(str(xt[i]) )
    outFile.write( '\n' )
outFile.write( 'SCALARS x_l float\n')
outFile.write( 'LOOKUP_TABLE x_l\n')  # longitudinal position
for i in range( n_points ):
    outFile.write(str(xl[i]) )
    outFile.write( '\n' )
outFile.write( 'SCALARS x_c float\n')
outFile.write( 'LOOKUP_TABLE x_c\n')  # circumferential position
for i in range( n_points ):
    outFile.write(str(xc[i]) )
    outFile.write( '\n' )
outFile.write( 'VECTORS e_c float\n')
for i in range( n_points ):
    ec_sel = e_c[i,:] / np.linalg.norm(e_c[i,:])
    for jj in range(3):
        outFile.write(str(ec_sel[jj])+' ' )
    outFile.write( '\n' )
outFile.write( 'VECTORS e_l float\n')
for i in range( n_points ):
    el_sel = e_l[i,:] / np.linalg.norm(e_l[i,:])
    for jj in range(3):
        outFile.write(str(el_sel[jj])+' ' )
    outFile.write( '\n' )
outFile.write( 'VECTORS e_t float\n')
for i in range( n_points ):
    ec_sel = e_c[i,:]
    el_sel = e_l[i,:]
    et_sel = np.cross(ec_sel,el_sel)
    if np.linalg.norm(et_sel) > 0:
        et_sel = et_sel / np.linalg.norm(et_sel)
    else:
        et_sel = [0.0,0.0,1.0]
    for jj in range(3):
        outFile.write(str(et_sel[jj])+' ' )
    outFile.write( '\n' )
outFile.close()