class class_FibersData:
    def __init__(self, endo_fiber_angle,epi_fiber_angle,endo_beta_angle,epi_beta_angle,gamma_angle):
        self.endo_fiber_angle = endo_fiber_angle
        self.epi_fiber_angle  = epi_fiber_angle
        self.endo_beta_angle  = endo_beta_angle
        self.epi_beta_angle   = epi_beta_angle
        self.gamma_angle      = gamma_angle

class matParameters:
    def __init__(self, a_iso, b_iso, a_f, b_f, a_s, b_s, a_fs, b_fs, k):
        self.a_iso = a_iso
        self.b_iso = b_iso
        self.a_f   = a_f
        self.b_f   = b_f
        self.a_s   = a_s
        self.b_s   = b_s
        self.a_fs  = a_fs
        self.b_fs  = b_fs
        self.k     = k

def LoadModelAnatomy(vtk_mean):

    import numpy as np
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_mean)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    data = reader.GetOutput()

    n_points = data.GetNumberOfPoints()
    n_el     = data.GetNumberOfCells()
    Coords   =  vtk_to_numpy(data.GetPoints().GetData())
    Els      = np.zeros((n_el,4),dtype=int)
    for i in range(n_el):
        cell_type = data.GetCellType(i)
        n_nodes_el   = data.GetCell(i).GetPointIds().GetNumberOfIds()
        for n_sel in range(n_nodes_el):
            Els[i,n_sel] = int(data.GetCell(i).GetPointId(n_sel))

    labels  = vtk_to_numpy(data.GetPointData().GetArray('labels'))
    x_c  = vtk_to_numpy(data.GetPointData().GetArray('x_c'))
    x_l  = vtk_to_numpy(data.GetPointData().GetArray('x_l'))
    x_t  = vtk_to_numpy(data.GetPointData().GetArray('x_t'))
    e_c  = vtk_to_numpy(data.GetPointData().GetVectors('e_c'))
    e_l  = vtk_to_numpy(data.GetPointData().GetVectors('e_l'))
    e_t  = vtk_to_numpy(data.GetPointData().GetVectors('e_t'))

    Node_par_coords = np.zeros((n_points,4))
    Node_par_coords[:,0] = labels
    Node_par_coords[:,1] = x_c
    Node_par_coords[:,2] = x_l
    Node_par_coords[:,3] = x_t

    faces_connectivity = np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
    Faces_Endo = []
    start_faces = True
    for kk in range(n_el):
        el_points = Els[kk,:]
        for jj in range(4):
            if all(labels[int(v)] == 2 for v in el_points[faces_connectivity[jj]]):
                if start_faces:
                    Faces_Endo  = np.array(el_points[faces_connectivity[jj]],dtype=int).reshape(1,-1)
                    start_faces = False
                else:
                    Faces_Endo = np.concatenate((Faces_Endo,np.array(el_points[faces_connectivity[jj]],dtype=int).reshape(1,-1)),0)

    return Coords, Els, n_points, n_el, Node_par_coords, e_t, e_l, e_c, Faces_Endo

def GenerateNodalAreas(Faces_Endo,Coords):

    import numpy as np
    
    Nodal_area  = np.zeros((Coords.shape[0],3))
    for jj in range(Faces_Endo.shape[0]):
        area_vector = 0.5*(np.cross(Coords[Faces_Endo[jj][1],:]-Coords[Faces_Endo[jj][0],:],Coords[Faces_Endo[jj][2],:]-Coords[Faces_Endo[jj][0],:]))
        for ll in range(3):
            Nodal_area[Faces_Endo[jj][ll],:] += area_vector/3.0
    return Nodal_area

def GradientOperator_AvgBased(Coords,Els,Node_par_coords):
    # Claudio Mancinellia , Marco Livesub  and Enrico Puppoa (2019),  
    #     A Comparison of Methods for Gradient Field Estimation on Simplicial Meshes
    #     in Computers & Graphics (80), 37-50, doi.org/10.1016/j.cag.2019.03.005 

    import numpy as np

    n_el     = Els.shape[0]
    n_points = Coords.shape[0]

    Vol_el       = np.zeros((n_el,1))
    Nodal_volume = np.zeros((n_points,1))

    dFcdx = np.zeros((n_el,n_points))
    dFcdy = np.zeros((n_el,n_points))
    dFcdz = np.zeros((n_el,n_points))

    dFdx = np.zeros((n_points,n_points))
    dFdy = np.zeros((n_points,n_points))
    dFdz = np.zeros((n_points,n_points))

    W = np.zeros((n_points,n_el))

    for sel_el in range(n_el):
        AA = np.zeros((3,3))
        AA[0,0] = Coords[Els[sel_el,1],0] - Coords[Els[sel_el,0],0]
        AA[0,1] = Coords[Els[sel_el,1],1] - Coords[Els[sel_el,0],1]
        AA[0,2] = Coords[Els[sel_el,1],2] - Coords[Els[sel_el,0],2]
        AA[1,0] = Coords[Els[sel_el,2],0] - Coords[Els[sel_el,0],0]
        AA[1,1] = Coords[Els[sel_el,2],1] - Coords[Els[sel_el,0],1]
        AA[1,2] = Coords[Els[sel_el,2],2] - Coords[Els[sel_el,0],2]
        AA[2,0] = Coords[Els[sel_el,3],0] - Coords[Els[sel_el,0],0]
        AA[2,1] = Coords[Els[sel_el,3],1] - Coords[Els[sel_el,0],1]
        AA[2,2] = Coords[Els[sel_el,3],2] - Coords[Els[sel_el,0],2]

        invA = np.linalg.inv(AA)

        dFcdx[sel_el,Els[sel_el,0]] = - np.sum(invA[0,:])
        dFcdx[sel_el,Els[sel_el,1]] = invA[0,0]
        dFcdx[sel_el,Els[sel_el,2]] = invA[0,1]
        dFcdx[sel_el,Els[sel_el,3]] = invA[0,2]

        dFcdy[sel_el,Els[sel_el,0]] = -np.sum(invA[1,:])
        dFcdy[sel_el,Els[sel_el,1]] = invA[1,0]
        dFcdy[sel_el,Els[sel_el,2]] = invA[1,1]
        dFcdy[sel_el,Els[sel_el,3]] = invA[1,2]

        dFcdz[sel_el,Els[sel_el,0]] = -np.sum(invA[2,:])
        dFcdz[sel_el,Els[sel_el,1]] = invA[2,0]
        dFcdz[sel_el,Els[sel_el,2]] = invA[2,1]
        dFcdz[sel_el,Els[sel_el,3]] = invA[2,2]

    # dFcdx, dFcdy,dFcdz are correct, validated with FEniCs and relative errors are around 10-5
    # only at apex they are around 1% (mesh distortion)

    # Volume weighted projection
    for i in range(n_el):
        Vol_el[i] = 1.0/6.0*abs((Coords[Els[i,3],:]-Coords[Els[i,0],:]).dot( np.cross(Coords[Els[i,2],:]-Coords[Els[i,0],:],Coords[Els[i,1],:]-Coords[Els[i,0],:])))
        for n_sel in Els[i]:
            Nodal_volume[n_sel] += Vol_el[i]/4.0

    for i in range(n_points):
        Els_per_node = np.where(Els == i)[0]

        for sel_el in Els_per_node: 
            dFdx[i,:] += dFcdx[sel_el,:].copy()*Vol_el[sel_el]/4.0/Nodal_volume[i]
            dFdy[i,:] += dFcdy[sel_el,:].copy()*Vol_el[sel_el]/4.0/Nodal_volume[i]
            dFdz[i,:] += dFcdz[sel_el,:].copy()*Vol_el[sel_el]/4.0/Nodal_volume[i]

    return dFcdx, dFcdy, dFcdz,dFdx, dFdy, dFdz, Nodal_volume,Vol_el

def LoadPODmodes_FunctionalModel(POD_folder,n_modes):

    import sys,os
    import numpy as np

    snapshots      = np.sort(next(os.walk(POD_folder))[2])
    n_snapshots    = len(snapshots)-1
    n_modes        = np.min([n_modes,n_snapshots])

    for m_sel in range(n_modes):
        Phi_matrix = np.load(POD_folder+'/Phi'+str(m_sel)+'_points.npy')
        if m_sel == 0:
            PHI = np.concatenate((Phi_matrix[:,0].reshape(-1,1),Phi_matrix[:,1].reshape(-1,1),Phi_matrix[:,2].reshape(-1,1)),0)
        else:
            pp_sel = np.concatenate((Phi_matrix[:,0].reshape(-1,1),Phi_matrix[:,1].reshape(-1,1),Phi_matrix[:,2].reshape(-1,1)),0)
            PHI    = np.concatenate((PHI,pp_sel),1)

    if os.path.isfile(POD_folder+'/Amplitudes_min_max.txt'):
        amplitudes = np.loadtxt(POD_folder+'/Amplitudes_min_max.txt')
        return PHI,n_modes, amplitudes[:n_modes,:]
    elif os.path.isfile(POD_folder+'/Amplitude_range.txt'):
        amplitudes = np.loadtxt(POD_folder+'/Amplitude_range.txt')
        return PHI,n_modes, amplitudes[:n_modes,:]
    else:
        return PHI,n_modes, 0.0

def GenerateFibers(e_t_vector,e_l_vector,e_c_vector,Node_par_coords,FiberData):

    import numpy as np

    n_points = Node_par_coords.shape[0]

    fx =  np.zeros((n_points,1))
    fy =  np.zeros((n_points,1))
    fz =  np.zeros((n_points,1))

    sx =  np.zeros((n_points,1))
    sy =  np.zeros((n_points,1))
    sz =  np.zeros((n_points,1))

    # Normalize fibers
    for i in range(n_points):
        x_c = Node_par_coords[i,1]
        x_l = Node_par_coords[i,2]
        x_t = Node_par_coords[i,3]
        e_c = e_c_vector[i,:]
        e_l = e_l_vector[i,:]
        e_t = e_t_vector[i,:]

        e_c = e_c/np.linalg.norm(e_c)
        e_t = np.cross(e_c,e_l)
        e_t = e_t/np.linalg.norm(e_t)
        e_l = e_l - np.dot(e_c,e_l)*e_c
        e_l = e_l/np.linalg.norm(e_l)

        alfa_angle  = np.pi/180.0*(FiberData.epi_fiber_angle*x_t + FiberData.endo_fiber_angle*(1.0 - x_t))
        gamma_angle = FiberData.gamma_angle*np.pi/180.0

        vf_dir   = np.cos(alfa_angle)*e_c + np.sin(alfa_angle)*e_l
        vf_dir   = vf_dir/np.linalg.norm(vf_dir)
        vs0_dir  = - np.cos(gamma_angle)*e_t + np.sin(gamma_angle)*e_l
        if np.linalg.norm(vs0_dir) > 0:
            vs_dir  = vs0_dir/np.linalg.norm(vs0_dir)
        else:
            vs_dir = vs_0_dir
        vs_dir = vs_dir - np.dot(vf_dir,vs_dir)*vf_dir
        vs_dir = vs_dir / np.linalg.norm(vs_dir)

        fx[i] = vf_dir[0]
        fy[i] = vf_dir[1]
        fz[i] = vf_dir[2]
        sx[i] = vs_dir[0]
        sy[i] = vs_dir[1]
        sz[i] = vs_dir[2]

    return fx[:,0],fy[:,0],fz[:,0],sx[:,0],sy[:,0],sz[:,0]

def WriteFibers2VTK(Coords,Els,fx,fy,fz,sx,sy,sz,out_file):
    import numpy as np

    outFile = open(out_file,'w')
    outFile.write('# vtk DataFile Version 4.0\n')
    outFile.write('vtk output\n')
    outFile.write('ASCII\n')
    outFile.write('DATASET UNSTRUCTURED_GRID \n')
    outFile.write('POINTS '+str(Coords.shape[0])+' float\n')
    for j in range(Coords.shape[0]):
        outFile.write(str(Coords[j,0])+' ')
        outFile.write(str(Coords[j,1])+' ')
        outFile.write(str(Coords[j,2])+' ')
        outFile.write('\n')
    outFile.write( 'CELLS ' + str( Els.shape[0] ) + ' ' + str( (Els.shape[0]) * 5 ) )
    outFile.write('\n')
    for k in range( Els.shape[0] ):
        outFile.write( '4 ' )
        for j in range( 4 ):
            outFile.write( str( Els[k,j]) + ' ' )
        outFile.write('\n')
    # write cell types
    outFile.write( '\n\nCELL_TYPES ' + str( Els.shape[0] ) )
    for k in range( Els.shape[0] ):
        outFile.write( '\n10' )
    outFile.write('\nPOINT_DATA '+str(Coords.shape[0])+'\n')
    outFile.write('VECTORS f float \n')
    for k in range(Coords.shape[0]):
        outFile.write(str(fx[k])+' '+str(fy[k])+' '+str(fz[k])+' '+'\n')
    outFile.write('VECTORS s float \n')
    for k in range(Coords.shape[0]):
        outFile.write(str(sx[k])+' '+str(sy[k])+' '+str(sz[k])+' '+'\n')
    outFile.close()
    return 0