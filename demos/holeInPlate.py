'''
A script to implement interpolation-based RKPM with FEniCSx 
Solving the hole in plate problem with low order geometric approximation, and double extraction

Author: Jennifer E Fromm
'''

import numpy as np 
from IntRKPM import common, linAlgHelp, classicRKPMUtils
import os.path
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, io, cpp
from dolfinx.fem import petsc
import ufl

from scipy import sparse

from timeit import default_timer

def eps(u):
    return ufl.sym(ufl.grad(u))
def sigma(eps):
    return 2.0*mu*eps+ lam*ufl.tr(eps)*ufl.Identity(2)
def eps_from_sig(sig):
    return (1/E)*((1+nu)*sig - nu*ufl.tr(sig)*ufl.Identity(2))

def sig_exact_ufl_bi(x): 
    # bi-axial stress 
    # a = hole radius
    # sig_top = applied tension at top 
    tol = 0.0001
    r_coord = (x[0]**2 + x[1]**2)**0.5
    r = ufl.max_value(r_coord,np.ones_like(r_coord)*R-tol)
    a_r = a/(r+tol)
    theta = ufl.atan(x[1]/(x[0]+ 1e-8))

    sig_rr = sig_top*(1 - a_r**2)
    sig_tt = sig_top*(1 + a_r**2)

    sig_polar = ufl.as_tensor([[sig_rr, 0],[0,sig_tt]])
    convert = ufl.as_tensor([[ufl.cos(theta), -ufl.sin(theta)],[ufl.sin(theta),ufl.cos(theta)]])
    sig_cart = ufl.dot(ufl.dot(convert,sig_polar),(convert.T))
    return sig_cart


def Left(x):
    return np.logical_and(np.isclose(x[0],0), np.greater_equal(x[1],R))
def Right(x):
    return np.isclose(x[0],L)
def Top(x):
    return np.isclose(x[1],L)
def Bottom(x):
    return np.logical_and(np.isclose(x[1],0), np.greater_equal(x[0],R))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--wf',dest='wf',default=False,
                    help='Write error to file (True/False)')
parser.add_argument('--vd',dest='vd',default=False,
                    help='visualize data by generating mesh files with solution, defualt False')
parser.add_argument('--of',dest='of',default="HIPResults.csv",
                    help='Output file destination')
parser.add_argument('--n',dest='n',default=1,
                    help='RK polynomial order')
parser.add_argument('--k',dest='k',default=None,
                    help='FG polynomial order, default to n')
parser.add_argument('--ref',dest='ref',default=0,
                    help='Refinement level')
parser.add_argument('--fgr',dest='fgr',default=0,
                    help='Option to add foreground refinement')
parser.add_argument('--lr',dest='lr',default=0,
                    help='Local foreground refinement (default = 0)')
parser.add_argument('--supp',dest='supp',default=3,
                    help='Support size, default 3')
parser.add_argument('--kernel',dest='kernel',default='SPLIN3',
                    help='Kernel type, default is SPLIN3')
parser.add_argument('--nnn',dest='nnn',default=4,
                    help='Number of nearest neighbors')
parser.add_argument('--st',dest='st',default='cir',
                    help='support type, circular (cir), rectangular (rec), or tensor product (TP)')
parser.add_argument('--eps',dest='eps',default=0.5,
                    help="Float between 0 and 1 controlling level of perturbation, default 0.5")
parser.add_argument('--genNew',dest='genNew',default=False,
                    help='Generate new extraction operator, even if file exists')
args = parser.parse_args()

write_file=args.wf
if write_file == 'True':
    write_file = True
else:
    write_file = False
visOutput=args.vd
if visOutput == 'True':
    visOutput = True
else:
    visOutput = False

genNew = args.genNew
if genNew == 'True':
    genNew = True
else:
    genNew = False
output_file = args.of
supportType = args.st

n = int(args.n)
nnn = int(args.nnn)
k = args.k 
if k is None:
    k = n 
else:
    k = int(k)

ref_val = int(args.ref)
fgr= int(args.fgr)
fg_ref = ref_val + fgr
lr_val = int(args.lr)

supp = n + 1
gridEps = float(args.eps)
kernel = args.kernel
#symmetric - sgn is positive, nonsymmetric- sgn is negative
sgn = 1 

ref_dir = 'FG'+str(lr_val)+'/R'+str(fg_ref)+'/'

fg_ref_diff = fg_ref - ref_val
opt = '_'
EX_MAT_DIR= ref_dir + "ExOps/n"+str(n)+"k"+str(k)+"fgs"+str(fg_ref_diff)+"/"+opt + "/"


saveExOp = True
comm = MPI.COMM_WORLD
L = 5
R = 1
aveKernel = L/ (8*2**ref_val)

# create RKPM basis 
RKPMBasis = classicRKPMUtils.RKPMBasisGenerator(ref_val)

t_file ="RKPMPoints/R"+str(ref_val)+"/tri.xdmf"
edgeNodes = 8
mesh_exists=os.path.isfile(t_file) 
if mesh_exists:
    RKPMBasis.readPoints(t_file)
else:
    RKPMBasis.makePoints(meshEps=gridEps,corners=[[0,0],[L,L]],edgeNodes=[edgeNodes,edgeNodes])
    #RKPMBasis.savePoints(t_file)

RKPMBasis.makeBasis(kernelType=kernel, polynomialOrder=n,supportSize=supp,numberNeighbors=nnn,supportType=supportType)


#mid ground (mg)
n_cells = (2**fgr)*(8)*(2**ref_val)
mg_domain = mesh.create_rectangle(comm, [np.array([0.0, 0.0]), np.array([L, L])],
                               [n_cells, n_cells], mesh.CellType.quadrilateral)
# mg_space
el = ufl.FiniteElement("DG", mg_domain.ufl_cell(),k)
V_mg = fem.FunctionSpace(mg_domain, el)
x_mg   = V_mg.tabulate_dof_coordinates()


MG_EX_MAT_FILE = EX_MAT_DIR + "mg_RK.dat"
t_start = default_timer()
readExOp=os.path.isfile(MG_EX_MAT_FILE) 
if readExOp and not genNew:
    print("reading in ExOp from file")
    M_mg = PETSc.Mat().create(MPI.COMM_WORLD)
    viewer = PETSc.Viewer(MPI.COMM_WORLD).createBinary(MG_EX_MAT_FILE,'r')
    M_mg = (M_mg.load(viewer))
    M_mg.assemble()
else:
    print('creating Ex Op')
    M_mg_scalar = common.createM(V_mg,RKPMBasis,nFields=1,returnAsSparse=True)
    nFields = 2
    M_lil_multiField =  sparse.lil_array(((nFields*M_mg_scalar.shape[0]),nFields*M_mg_scalar.shape[1]))
    M_lil_multiField[0:(M_mg_scalar.shape[0]),0:(M_mg_scalar.shape[1])] = M_mg_scalar
    M_lil_multiField[M_mg_scalar.shape[0]:(2*M_mg_scalar.shape[0]),M_mg_scalar.shape[1]:2*(M_mg_scalar.shape[1])] = M_mg_scalar
    M_csr = M_lil_multiField.tocsr()
    M_mg = PETSc.Mat().createAIJ(size=M_csr.shape,csr=(M_csr.indptr, M_csr.indices,M_csr.data))
    M_mg.assemble()

    """
    Note: it is highly recomended to set up the directory structure to save and reuse extraction operators
    print('Saving mid-ground ExOp to file')
    viewer = PETSc.Viewer(MPI.COMM_WORLD).createBinary(MG_EX_MAT_FILE, 'w')
    viewer(M_mg)
    """

t_stop = default_timer()
t_mg_ex = t_stop-t_start

nu = 0.3
E = 200e9
sig_top = 1000000
a = R

lam = E*nu/((1+nu)*(1-nu))
K = E/(3 *(1 -2*nu))
mu =(3/2)*(K-lam)

#Nitsches penalty parameter
beta = 10*mu

# cell markers, from mesh file
inside_ID = 0
outside_ID = 1

# facet markers, user specified
top_ID = 0 
bottom_ID = 1
left_ID = 2
right_ID = 3
interface_ID = 4

mesh_types = ["tri","quad"]
Vs = []
As =[]
bs = []
Ms = []
domains = []
sig_exs =[]
eps_exs = []
us = []
t_exs = []
dxs = []
fg_dofs = []
omega0vs = [] 

facet_markers = [top_ID, bottom_ID, left_ID, right_ID]
facet_functions = [Top, Bottom, Left, Right]

num_facet_phases =len(facet_markers)

for subMeshType in mesh_types:
    meshFile = ref_dir + "meshes/" + subMeshType + "_materials.xdmf"
    with io.XDMFFile(MPI.COMM_WORLD, meshFile, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid",ghost_mode=cpp.mesh.GhostMode.shared_facet)
        ct = xdmf.read_meshtags(domain, name="Grid")
        xdmf.close()
    cell_mat = ct.values

    Omega0_cells = ct.find(inside_ID)
    Omega1_cells  = ct.find(outside_ID)

    dim = domain.topology.dim
    domain.topology.create_connectivity(dim-1, dim)

    num_facets = domain.topology.index_map(dim-1).size_global

    f_to_c_conn = domain.topology.connectivity(dim-1,dim)

    interface_facets = []
    interface_facet_marks = []
    for facet in range(num_facets):
        marker = 0
        cells = f_to_c_conn.links(facet)
        for cell in cells:
            marker = marker + cell_mat[cell] +1
        if marker == 3: 
            interface_facets += [facet]
            interface_facet_marks += [interface_ID]
    interface_facets = np.asarray(interface_facets,dtype=np.int32)
    interface_facet_marks = np.asarray(interface_facet_marks,dtype=np.int32)
    
    # mark exterior boundaries using FEniCS function
    num_facet_phases =len(facet_markers)
    facets = np.asarray([],dtype=np.int32)
    facets_mark = np.asarray([],dtype=np.int32)
    for phase in range(num_facet_phases):
        facets_phase = mesh.locate_entities(domain, domain.topology.dim-1, facet_functions[phase])
        facets_phase_mark = np.full_like(facets_phase, facet_markers[phase])
        facets= np.hstack((facets,facets_phase))
        facets_mark = np.hstack((facets_mark,facets_phase_mark))

    # add interface facets to list
    facets= np.hstack((facets,interface_facets))
    facets_mark = np.hstack((facets_mark,interface_facet_marks))
    sorted_facets = np.argsort(facets)

    ft = mesh.meshtags(domain,dim-1,facets[sorted_facets], facets_mark[sorted_facets])

    #create weight function to use for visualization
    Q = fem.FunctionSpace(domain, ("DG", 0))
    Omega0 = fem.Function(Q)
    Omega0.x.array[Omega0_cells] = 1
    Omega0.x.array[Omega1_cells] = 0
    Omega0.x.scatter_forward()
    Omega1 = fem.Function(Q)
    Omega1.x.array[Omega0_cells] = 0
    Omega1.x.array[Omega1_cells] = 1
    Omega1.x.scatter_forward()

    Omega_all0 = fem.Function(Q)
    Omega_all0.x.array[Omega0_cells] = 0
    Omega_all0.x.array[Omega1_cells] = 0 
    Omega_all0.x.scatter_forward()



    # define integration measurements for the domain of interest and the interior surface of interest
    dx = ufl.Measure('dx',domain=domain,subdomain_data=ct,metadata={'quadrature_degree': 2*k})
    ds = ufl.Measure("ds",domain=domain,subdomain_data=ft,metadata={'quadrature_degree': 2*k})
    ds_exterior = ds(right_ID) + ds(left_ID) + ds(top_ID) + ds(bottom_ID)
    dS = ufl.Measure("dS",domain=domain,subdomain_data=ft,metadata={'quadrature_degree': 2*k})

    el = ufl.FiniteElement("DG", domain.ufl_cell(),k)
    mel = ufl.MixedElement([el, el])
    V = fem.FunctionSpace(domain, mel)
    V0, V0_to_V = V.sub(0).collapse()
    V1, V1_to_V = V.sub(1).collapse()
    V_flux =  fem.FunctionSpace(domain, ("DG", k-1))

    
    omega0V = fem.Function(V0)
    omega0V.interpolate(Omega0)
    omega1V = fem.Function(V0)
    omega1V.interpolate(Omega1)

    omegaall0V = fem.Function(V0)
    omegaall0V.interpolate(Omega_all0)


    u = fem.Function(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    eps_u = eps(u)
    sigma_u = sigma(eps_u)

    eps_v = eps(v)
    sigma_v = sigma(eps_v)

    sig_ex = sig_exact_ufl_bi(x)

    h = 0.1*(2**-ref_val)
    n_vec = ufl.FacetNormal(domain)
    res_u =ufl.inner(eps_v,sigma_u)*(dx(outside_ID))
    res_traction_t =-ufl.dot(ufl.dot(sig_ex,n_vec),v)*ds(top_ID)
    res_traction_r =-ufl.dot(ufl.dot(sig_ex,n_vec),v)*ds(right_ID)

    ds_sym = ds(left_ID) + ds(bottom_ID)

    nitsches_term = -sgn*ufl.dot(ufl.dot(ufl.dot(sigma_v,n_vec),n_vec),ufl.dot(u,n_vec))*ds_sym - ufl.dot(ufl.dot(ufl.dot(sigma_u,n_vec),n_vec),ufl.dot(v,n_vec))*ds_sym
    penalty_term = beta*(h**(-1))*ufl.dot(ufl.dot(u,n_vec),ufl.dot(v,n_vec))*ds_sym
    res_sym = penalty_term + nitsches_term 

    res = res_u + res_traction_t + res_traction_r + res_sym

    J = ufl.derivative(res,u)
    res_form = fem.form(res)
    res_petsc = fem.petsc.assemble_vector(res_form)
    J_form = fem.form(J)
    J_petsc = fem.petsc.assemble_matrix(J_form)

    J_petsc.assemble()

    t_start = default_timer()
    EX_MAT_FILE = EX_MAT_DIR + subMeshType + ".dat"
    readExOp=os.path.isfile(EX_MAT_FILE) 
    if readExOp and not genNew:
        print("reading in ExOp from file")
        M = PETSc.Mat().create(MPI.COMM_WORLD)
        viewer = PETSc.Viewer(MPI.COMM_WORLD).createBinary(EX_MAT_FILE,'r')
        M = (M.load(viewer))
        M.assemble()
    else:
        print('creating Ex Op')

        nFields = 2
        M_u0 = common.interpolation_matrix_nonmatching_meshes(V_mg, V0)
        M_u1 = common.interpolation_matrix_nonmatching_meshes(V_mg, V1)
        shape = M_u0.shape
        M_lil_multiField =  sparse.lil_array(((nFields*shape[0]),nFields*shape[1]))
        M_lil_multiField[V0_to_V, 0:shape[1]] = M_u0
        M_lil_multiField[V1_to_V,shape[1]:(2*shape[1])] = M_u1
        M_csr = M_lil_multiField.tocsr()
        M = PETSc.Mat().createAIJ(size=M_csr.shape,csr=(M_csr.indptr, M_csr.indices,M_csr.data))
        M.assemble()

        """
        Note: it is highly recomended to set up the directory structure to save and reuse extraction operators
        print('Saving ExOp to file')
        viewer = PETSc.Viewer(MPI.COMM_WORLD).createBinary(EX_MAT_PATH, 'w')
        viewer(M)
        """
    t_stop = default_timer()
    t_ex = t_stop-t_start

    [fg_dof,bg_dof] = M.getSize()
    A,b = linAlgHelp.assembleLinearSystemBackground(J_petsc,-res_petsc,M)
    
    Vs += [V]
    As += [A]
    bs += [b]
    Ms += [M]
    domains +=[domain]
    sig_exs +=[sig_ex]
    us += [u]
    dxs += [dx(outside_ID)] 
    t_exs += [t_ex]
    fg_dofs +=[fg_dof]
    omega0vs += [omega0V]


A_tri,A_quad = As
b_tri,b_quad = bs
M_tri,M_quad = Ms
domain_tri,domain_quad = domains
sig_ex_tri,sig_ex_quad = sig_exs
u_tri,u_quad = us
dx_tri,dx_quad = dxs
t_ex_tri,t_ex_quad = t_exs
fg_tri,fg_quad = fg_dofs

# add the two matrices and vectors
A_tri.axpy(1.0,A_quad)
b_tri.axpy(1.0,b_quad)

x = A_tri.createVecLeft()


t_start = default_timer()

# second level of extraction 
A_RK,b_RK = linAlgHelp.assembleLinearSystemBackground(A_tri,b_tri,M_mg)
x_RK = A_RK.createVecLeft()
linAlgHelp.solveKSP(A_RK,b_RK,x_RK,monitor=False,method='mumps')

t_stop = default_timer()
t_solve = t_stop-t_start

u_mg = A_tri.createVecLeft()
M_mg.mult(x_RK, u_mg)
common.transferToForeground(u_tri, u_mg, M_tri)
common.transferToForeground(u_quad, u_mg, M_quad)

# stress error norm
sig_tri = sigma(eps(u_tri))
L2_error = fem.assemble_scalar(fem.form(ufl.inner(sig_tri - sig_ex_tri , sig_tri - sig_ex_tri) * dx_tri))
error_L2_tri  = np.sqrt(domain.comm.allreduce(L2_error, op=MPI.SUM))
L2 = fem.assemble_scalar(fem.form(ufl.inner(sig_ex_tri,sig_ex_tri) * dx_tri))
L2_tri  = np.sqrt(domain.comm.allreduce(L2, op=MPI.SUM))

sig_quad = sigma(eps(u_quad))
L2_error = fem.assemble_scalar(fem.form(ufl.inner(sig_quad - sig_ex_quad,sig_quad - sig_ex_quad) * dx_quad))
error_L2_quad  = np.sqrt(domain.comm.allreduce(L2_error, op=MPI.SUM))

L2= fem.assemble_scalar(fem.form(ufl.inner(sig_ex_quad , sig_ex_quad ) * dx_quad))
L2_quad  = np.sqrt(domain.comm.allreduce(L2, op=MPI.SUM))

tri_L2_norm = error_L2_tri/ L2_tri

quad_L2_norm = error_L2_quad/ L2_quad

net_L2_norm = (error_L2_quad + error_L2_tri) / (L2_quad + L2_tri)

actualNP = RKPMBasis.nP
dofs_total = fg_tri + fg_quad

print(f"Stress Error : {net_L2_norm}")
print(f"Stress Error (tris): {tri_L2_norm}")
print(f"Stress Error (quads): {quad_L2_norm}")
print(f"Extraction Time (tris): {t_ex_tri}")
print(f"Extraction Time (quads): {t_ex_quad}")
print(f"Solver Time : {t_solve}")

#ref,n,L2,H1,kernel,supp,k,fg_r,t_ex_t,t_ex_q,t_solve
if write_file: 
    f = open(output_file,'a')
    f.write("\n")
    fs = str(actualNP)+","+str(dofs_total)+","+str(ref_val)+","+str(n)+","+str(net_L2_norm)+","\
        + str(tri_L2_norm)+"," + str(quad_L2_norm)+","\
        + kernel+","+str(supp) +","+ str(k)+","+ str(lr_val)+","+ str(fg_ref_diff)+","+ \
            str(t_ex_tri)+","+str(t_ex_quad)+","+str(t_solve)
    f.write(fs)
    f.close()



u_solns = [u_tri,u_quad]
if k > 1:
    strainFileWriter = common.outputVTX
else: 
    strainFileWriter = common.outputXDMF


if visOutput:
    i = 0 
    for subMeshType in mesh_types:
        V = Vs[i] 
        u = u_solns[i] 
        sig_ex = sig_exs[i]
        domain = domains[i]
        omega0v = omega0vs[i]
        
        
        folder = EX_MAT_DIR + "HIP_biAx/"

        with io.VTXWriter(domain.comm, folder+"material"+subMeshType+".bp", [omega0v], engine="BP4") as vtx:
            vtx.write(0.0)

        # plotting 
        U0, U0_to_W = V.sub(0).collapse()
        U1, U1_to_W = V.sub(1).collapse()

        u0_plot = fem.Function(U0)
        u1_plot = fem.Function(U1)

        u0_plot.x.array[:] = u.x.array[U0_to_W]
        u0_plot.x.scatter_forward()

        u1_plot.x.array[:] = u.x.array[U1_to_W]
        u1_plot.x.scatter_forward()   

        ur = ufl.sqrt(u[0]**2 + u[1]**2)
        ur_expr  = fem.Expression(ur,U0.element.interpolation_points())
        ur_plot = fem.Function(U0)
        ur_plot.interpolate(ur_expr)

        with io.VTXWriter(domain.comm, folder+"u0_"+subMeshType+".bp", [u0_plot], engine="BP4") as vtx:
            vtx.write(0.0)
        with io.VTXWriter(domain.comm, folder+"u1_"+subMeshType+".bp", [u1_plot], engine="BP4") as vtx:
            vtx.write(0.0)
        with io.VTXWriter(domain.comm, folder+"ur_"+subMeshType+".bp", [ur_plot], engine="BP4") as vtx:
            vtx.write(0.0)
        
        # plot strain 
        V_strain = fem.FunctionSpace(domain, ("DG", k-1))



        eps_soln = eps(u)
        folder_ecomp = folder + "strain_components/"
        

        strainFileWriter(eps_soln[0,0],V_strain,folder_ecomp,"e00_"+subMeshType)
        strainFileWriter(eps_soln[1,0],V_strain,folder_ecomp,"e10_"+subMeshType)
        strainFileWriter(eps_soln[0,1],V_strain,folder_ecomp,"e01_"+subMeshType)
        strainFileWriter(eps_soln[1,1],V_strain,folder_ecomp,"e11_"+subMeshType)
        eps_sol_mag = ufl.sqrt(ufl.inner(eps_soln,eps_soln)) 
        strainFileWriter(eps_sol_mag,V_strain,folder,"eps_mag_"+subMeshType)
        strainFileWriter(omega0v,V_strain,folder,"strainMat"+subMeshType)
        eps_ex = eps_from_sig(sig_ex)
        strainFileWriter(eps_ex[0,0],V_strain,folder_ecomp,"e00_ex_"+subMeshType)
        strainFileWriter(eps_ex[1,0],V_strain,folder_ecomp,"e10_ex_"+subMeshType)
        strainFileWriter(eps_ex[0,1],V_strain,folder_ecomp,"e01_ex_"+subMeshType)
        strainFileWriter(eps_ex[1,1],V_strain,folder_ecomp,"e11_ex_"+subMeshType)
        eps_ex_mag = ufl.sqrt(ufl.inner(eps_ex,eps_ex)) 
        strainFileWriter(eps_ex_mag,V_strain,folder,"eps_mag_ex_"+subMeshType)

        sig_soln = sigma(eps(u))
        folder_scomp = folder + "stress_components/"

        strainFileWriter(sig_soln[0,0],V_strain,folder_scomp,"s00_"+subMeshType)
        strainFileWriter(sig_soln[1,0],V_strain,folder_scomp,"s10_"+subMeshType)
        strainFileWriter(sig_soln[0,1],V_strain,folder_scomp,"s01_"+subMeshType)
        strainFileWriter(sig_soln[1,1],V_strain,folder_scomp,"s11_"+subMeshType)
        sig_sol_mag = ufl.sqrt(ufl.inner(sig_soln,sig_soln)) 
        strainFileWriter(sig_sol_mag,V_strain,folder,"sig_mag_"+subMeshType)

        
        strainFileWriter(sig_ex[0,0],V_strain,folder_scomp,"s00_ex_"+subMeshType)
        strainFileWriter(sig_ex[1,0],V_strain,folder_scomp,"s10_ex_"+subMeshType)
        strainFileWriter(sig_ex[0,1],V_strain,folder_scomp,"s01_ex_"+subMeshType)
        strainFileWriter(sig_ex[1,1],V_strain,folder_scomp,"s11_ex_"+subMeshType)
        sig_ex_mag = ufl.sqrt(ufl.inner(sig_ex,sig_ex)) 
        strainFileWriter(sig_ex_mag,V_strain,folder,"sig_mag_ex_"+subMeshType)

        i += 1
