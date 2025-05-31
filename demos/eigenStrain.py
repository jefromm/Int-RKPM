'''
A script to implement interpolation-based RKPM with FEniCSx 
Solving the eigenstrain problem with low order geometric approximation, double extraction and enrichment

Author: Jennifer E Fromm
'''



import numpy as np 
from IntRKPM import common, linAlgHelp, classicRKPMUtils
import os.path
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, io, geometry, cpp
from dolfinx.fem import petsc
import ufl

from timeit import default_timer
from scipy import sparse, spatial


def epsU(u):
    return ufl.sym(ufl.grad(u))
def sigma(eps):
    return 2.0*mu*eps+ lam*ufl.tr(eps)*ufl.Identity(2)


def u_exact_ufl_in(x): 
    r = (x[0]**2 + x[1]**2)**0.5
    theta = ufl.atan(x[1]/(x[0]+ 1e-8))
    ur = C1*r
    ux = ur*ufl.cos(theta)
    uy = ur*ufl.sin(theta)
    return ufl.as_vector([ux,uy])

def u_exact_ufl_BC(x): 
    r = ufl.operators.max_value(R/2,(x[0]**2 + x[1]**2)**0.5)
    theta = ufl.atan(x[1]/(x[0]+ 1e-8))
    ur = C1*R*R/(r)
    ux = ur*ufl.cos(theta)
    uy = ur*ufl.sin(theta)
    return ufl.as_vector([ux,uy])

def ur_exact_ufl_in(x): 
    r = (x[0]**2 + x[1]**2)**0.5
    ur = C1*r
    return ur

def ur_exact_ufl_BC(x): 
    r = ufl.operators.max_value(R/2,(x[0]**2 + x[1]**2)**0.5)
    ur = C1*R*R/(r)
    return ur


def custom_avg(val, param, domain, h = None):
    if h is not None:
        w_mag = h/param('+') + h/param('-')
        ave = (val('+')*h/param('+') + val('-')*h/param('-'))/w_mag 
    else:
        size = ufl.JacobianDeterminant(domain)
        h = size**(0.5)
        w_mag = h('+')/param('+') + h('-')/param('-')
        ave = (val('+')*h('+')/param('+') + val('-')*h('-')/param('-'))/w_mag 
    return ave 

def gamma_int(c, param, domain, h = None):
    if h is not None:
        w_mag = h/param('+') + h/param('-')
        ave = 2*(c*h + c*h)/w_mag  
    else:
        size = ufl.JacobianDeterminant(domain)
        h = size**(0.5)
        w_mag = h('+')/param('+') + h('-')/param('-')
        ave = 2*(c*h('+') + c*h('-'))/w_mag 
    return ave 
    

def interface_u(u,v,domain,dS,jump,weight,h, C_u=10,eps0=None):
    # specify normal directed away from inner section
    n = ufl.avg(weight*ufl.FacetNormal(domain))
    if eps0 == None:
        sig_u = sigma(epsU(u))
    else:
        sig_u = sigma(epsU(u)- eps0)
    sig_v = sigma(epsU(v))
    const = ufl.inner(ufl.avg(jump*u),ufl.dot(custom_avg((sig_v),E,domain),n))*dS
    adjconst = ufl.inner(ufl.avg(jump*v),ufl.dot(custom_avg((sig_u),E,domain),n))*dS
    gamma = gamma_int(C_u, E, domain,h=h)
    pen = gamma*ufl.inner(ufl.avg(jump*u),ufl.avg(jump*v))*dS
    return const - adjconst + pen 


def dirichlet_u(u,v,ud,domain,ds,h,sgn=-1,eps0=None,C_u=10):
    n = ufl.FacetNormal(domain)
    if eps0 == None:
        sig_u = sigma(epsU(u))
    else:
        sig_u = sigma(epsU(u)- eps0)
    sig_v = sigma(epsU(v))
    const = sgn*ufl.inner(ufl.dot(sig_v, n), (u-ud))*ds
    adjconst = ufl.inner(ufl.dot(sig_u, n), v)*ds
    gamma = C_u *E /h 
    pen = gamma*ufl.inner(v,(u-ud))*ds
    return  pen -const - adjconst 

def symmetry_u(u,v,g,domain,ds,h,sgn=-1,C_u=10,eps0 = None):
    beta = C_u*mu
    n = ufl.FacetNormal(domain)
    if eps0 == None:
        sig_u = sigma(epsU(u))
    else:
        sig_u = sigma(epsU(u)- eps0)
    sig_v = sigma(epsU(v))
    nitsches_term =  -sgn*ufl.dot(ufl.dot(ufl.dot(sig_v,n),n),(ufl.dot(u,n)- g))*ds - ufl.dot(ufl.dot(ufl.dot(sig_u,n),n),ufl.dot(v,n))*ds
    penalty_term = beta*(h**(-1))*ufl.dot((ufl.dot(u,n)-g),ufl.dot(v,n))*ds
    return nitsches_term + penalty_term 
    
def Left(x):
    return np.isclose(x[0],0)
def Right(x):
    return np.isclose(x[0],L)
def Top(x):
    return np.isclose(x[1],L)
def Bottom(x):
    return np.isclose(x[1],0)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--wf',dest='wf',default=False,
                    help='Write error to file (True/False)')
parser.add_argument('--vd',dest='vd',default=False,
                    help='visualize data by generating mesh files with solution, defualt False')
parser.add_argument('--of',dest='of',default="reproducingCond.csv",
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
parser.add_argument('--opt',dest='opt',default='enrich',
                    help='enrichment option to use- default is with enrichment ')
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
opt = args.opt

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

#supp = n + 1
supp = 3
gridEps = float(args.eps)
kernel = args.kernel


ref_dir = 'FG'+str(lr_val)+'/R'+str(fg_ref)+'/'

fg_ref_diff = fg_ref - ref_val
EX_MAT_DIR= ref_dir + "ExOps/n"+str(n)+"k"+str(k)+"fgs"+str(fg_ref_diff)+"/"+opt + "/"


saveExOp = True
comm = MPI.COMM_WORLD
L = 5
R = 1
aveKernel = L/ (8*2**ref_val)
# cell markers, from mesh file
inside_ID = 0
outside_ID = 1

# facet markers, user specified
top_ID = 0 
bottom_ID = 1
left_ID = 2
right_ID = 3
interface_ID = 4

if opt == '_':
    print( 'No enrichment used')
    enrich=False
elif opt == 'enrich':
    print( 'Using Heaviside enrichment')
    enrich = True
else:
    print("scaling option not supported")
    exit() 


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

# identify mg cells (and basis functions) for enrichement 
# use short hand, of identify the cells covered by the tri- mesh 
meshFile = ref_dir + "meshes/tri_materials.xdmf"
with io.XDMFFile(MPI.COMM_WORLD, meshFile, "r") as xdmf:
    tri_domain = xdmf.read_mesh(name="Grid",ghost_mode=cpp.mesh.GhostMode.shared_facet)
    xdmf.close()

V_tri = fem.FunctionSpace(tri_domain, ("DG", 0))
x_tri   = V_tri.tabulate_dof_coordinates()

bb_tree         = geometry.bb_tree(mg_domain, mg_domain.topology.dim)
cell_candidates = geometry.compute_collisions_points(bb_tree, x_tri)
cells           = []
colliding_cells = geometry.compute_colliding_cells(mg_domain, cell_candidates, x_tri)

for i, point in enumerate(x_tri):
    for j in range(len(colliding_cells.links(i))):
        cells.append(colliding_cells.links(i)[j])
cells_          = np.unique(np.array(cells))
cell_dofs = []

for nn in range(0,len(cells_)):
    for nnn in range(len(V_mg.dofmap.cell_dofs(cells_[nn]))):
        cell_dofs.append(V_mg.dofmap.cell_dofs(cells_[nn])[nnn])

cell_dofs = np.unique(np.asarray(cell_dofs,dtype = np.int32))
mg_e_dof_coords = x_mg[cell_dofs,0:2]


actualNP = RKPMBasis.nP
if enrich:
    kernelRads = RKPMBasis.domainKernel
    numNodes = len(kernelRads)
    eBasisList = np.ones((numNodes, 2))*-1
    
    
    dd,_ = spatial.KDTree(mg_e_dof_coords).query(RKPMBasis.nodeCoords)
     
    enrich_ids = np.asarray(np.nonzero(dd <= kernelRads))[0]
    num_eRKdofs = enrich_ids.shape[0]
    new_eRKs = np.arange(numNodes,numNodes+num_eRKdofs)
    eBasisList[enrich_ids,0] = enrich_ids
    eBasisList[enrich_ids,1] = new_eRKs
    
    newNodeNum = numNodes+num_eRKdofs
    actualNP = newNodeNum


num_mg_dofs = V_mg.dofmap.index_map.size_global
new_mg_dofs = cell_dofs.shape[0]
new_mg_dofs_net = num_mg_dofs + new_mg_dofs
mg_eBasisList = np.ones((num_mg_dofs, 2))*-1
mg_nextNodeID = num_mg_dofs
for mg_dof in cell_dofs:
    mg_eBasisList[mg_dof,0] = mg_dof
    mg_eBasisList[mg_dof,1] = mg_nextNodeID 
    mg_nextNodeID += 1 



# create mg enrichment functions 
# first find coarse dofs in each camp with cover from quad mesh 
meshFile = ref_dir + "meshes/quad_materials.xdmf"
with io.XDMFFile(MPI.COMM_WORLD, meshFile, "r") as xdmf:
    quad_domain = xdmf.read_mesh(name="Grid",ghost_mode=cpp.mesh.GhostMode.shared_facet)
    ct = xdmf.read_meshtags(quad_domain, name="Grid")
    xdmf.close()
cell_mat = ct.values
Omega0_cells = ct.find(inside_ID)
Omega1_cells  = ct.find(outside_ID)
Q = fem.FunctionSpace(quad_domain, ("DG", 0))

x_quad   = Q.tabulate_dof_coordinates()
x_Omega0 = x_quad[Omega0_cells,:]

cell_candidates_Omega0 = geometry.compute_collisions_points(bb_tree, x_Omega0)
cells_Omega0           = []
colliding_cells_Omega0 = geometry.compute_colliding_cells(mg_domain, cell_candidates_Omega0, x_Omega0)

for i, point in enumerate(x_Omega0):
    for j in range(len(colliding_cells_Omega0.links(i))):
        cells_Omega0 .append(colliding_cells_Omega0.links(i)[j])
cells_Omega0_ = np.unique(np.array(cells_Omega0))

cell_dofs_Omega0 = []
for nn in range(0,len(cells_Omega0_)):
    for nnn in range(len(V_mg.dofmap.cell_dofs(cells_Omega0_[nn]))):
        cell_dofs_Omega0.append(V_mg.dofmap.cell_dofs(cells_Omega0_[nn])[nnn])

### repeat for Omega1
x_Omega1 = x_quad[Omega1_cells,:]
cell_candidates_Omega1 = geometry.compute_collisions_points(bb_tree, x_Omega1)
cells_Omega1          = []
colliding_cells_Omega1 = geometry.compute_colliding_cells(mg_domain, cell_candidates_Omega1, x_Omega1)

for i, point in enumerate(x_Omega1):
    for j in range(len(colliding_cells_Omega1.links(i))):
        cells_Omega1 .append(colliding_cells_Omega1.links(i)[j])
cells_Omega1_ = np.unique(np.array(cells_Omega1))

cell_dofs_Omega1 = []
for nn in range(0,len(cells_Omega1_)):
    for nnn in range(len(V_mg.dofmap.cell_dofs(cells_Omega1_[nn]))):
        cell_dofs_Omega1.append(V_mg.dofmap.cell_dofs(cells_Omega1_[nn])[nnn])

Omega0_mg_IDs  = np.ones(new_mg_dofs_net)*-1
Omega0_mg_IDs[cell_dofs_Omega0] = np.ones_like(cell_dofs_Omega0)
Omega0_mg_IDs[cell_dofs_Omega1] = np.zeros_like(cell_dofs_Omega1)

Omega1_mg_IDs  = np.ones(new_mg_dofs_net)*-1
Omega1_mg_IDs[cell_dofs_Omega0] = np.zeros_like(cell_dofs_Omega0)
Omega1_mg_IDs[cell_dofs_Omega1] = np.ones_like(cell_dofs_Omega1)

# now deal with enriched dofs - original ID become Omega0, new ID becomes Omega1 
Omega0_mg_IDs[cell_dofs] = np.ones_like(cell_dofs)
Omega1_mg_IDs[cell_dofs] = np.zeros_like(cell_dofs)

Omega0_mg_IDs[num_mg_dofs:new_mg_dofs_net] = np.zeros_like(np.arange(num_mg_dofs,new_mg_dofs_net))
Omega1_mg_IDs[num_mg_dofs:new_mg_dofs_net] = np.ones_like(np.arange(num_mg_dofs,new_mg_dofs_net))

enrich_vecs = [Omega0_mg_IDs,Omega1_mg_IDs]

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
    if enrich:
        M_mg_scalar = common.createM(V_mg,RKPMBasis,nFields=1,returnAsSparse=True)
        M_lil =  sparse.lil_array(((new_mg_dofs_net),(newNodeNum)))
        # first add in original matrix 
        M_lil[np.arange(0,num_mg_dofs),0:RKPMBasis.nP] = M_mg_scalar
        # now, add in RK evals at new mg nodes
        M_lil[np.arange(num_mg_dofs,new_mg_dofs_net),0:RKPMBasis.nP] = M_mg_scalar[cell_dofs,:]
        # need to make a copy of M_lil pre enrichment 
        M_lil_pre_enrich = M_lil[:,(eBasisList[:,0] >= 0 )].toarray()

        # next, duplicate columns for RK node 
        for matID in range(eBasisList.shape[1]):
            M_lil[:,(eBasisList[:,matID][eBasisList[:,matID] >= 0 ])] =M_lil_pre_enrich*np.atleast_2d(enrich_vecs[matID]).T
            
        nFields = 2

        M_lil_multiField =  sparse.lil_array(((nFields*new_mg_dofs_net),nFields*newNodeNum))
        M_lil_multiField[0:(new_mg_dofs_net),0:(newNodeNum)] = M_lil
        M_lil_multiField[new_mg_dofs_net:(2*new_mg_dofs_net),newNodeNum:2*(newNodeNum)] = M_lil
        M_csr = M_lil_multiField.tocsr()
        
        M_mg = PETSc.Mat().createAIJ(size=M_csr.shape,csr=(M_csr.indptr, M_csr.indices,M_csr.data))
        M_mg.assemble()

    else:
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

mu_inside = 390.63
mu_outside = 338.35
lam_inside = 497.16
lam_outside = 656.79
eps0 = 0.1
C1 = (lam_inside + mu_inside)*eps0/(lam_inside + mu_inside + mu_outside) 

mesh_types = ["tri","quad"]
Vs = []
As =[]
bs = []
Ms = []
domains = []
u_exs =[]
ur_exs = []
us = []
t_exs = []
dxs = []
mus = []
lams = []


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
        #cells = [f_to_c_conn[2*facet], f_to_c_conn[2*facet + 1]]
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

    #create weight function to control integration on the interior surface 
    Q = fem.FunctionSpace(domain, ("DG", 0))
    Omega0 = fem.Function(Q)
    Omega0.x.array[Omega0_cells] = 1
    Omega0.x.array[Omega1_cells] = 0
    Omega0.x.scatter_forward()
    Omega1 = fem.Function(Q)
    Omega1.x.array[Omega0_cells] = 0
    Omega1.x.array[Omega1_cells] = 1
    Omega1.x.scatter_forward()

    # create DG function to calculate the jump 
    # Using the notation from (Schmidt 2023), the interior is material m and the exterior is n 
    # jump == [[.]] = (.)^m - (.)^n 
    jump = 2*Omega0 - 2*Omega1
    w_0 = 2*Omega0
    w_1 = 2*Omega1

    mu = Omega0*mu_inside +Omega1* mu_outside
    lam = Omega0*lam_inside +Omega1*lam_outside
    E = mu*(3*lam+2*mu)/(lam+mu) 

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
    enrichFuncs = [omega0V,omega1V]

    u = fem.Function(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    u_ex = fem.Function(V)
    u_exact_in = u_exact_ufl_in(x) 
    u_exact_BC = u_exact_ufl_BC(x) 
    ur_exact_in = ur_exact_ufl_in(x) 
    ur_exact_BC = ur_exact_ufl_BC(x) 
    eps0_tensor = eps0* ufl.as_tensor([[1,0],[0,1]])*Omega0
    u_exact = u_exact_in*Omega0 + u_exact_BC*Omega1
    ur_exact = ur_exact_in*Omega0 + ur_exact_BC*Omega1
    
    

    h = 0.1*(2**-ref_val)
    epsE = epsU(u) - eps0_tensor
    res_u =ufl.inner(epsU(v),sigma(epsE))*(dx(inside_ID) + dx(outside_ID))
    resD_u_r = dirichlet_u(u,v,u_exact_BC,domain,ds(right_ID),h,eps0=eps0_tensor)
    resD_u_t = dirichlet_u(u,v,u_exact_BC,domain,ds(top_ID),h,eps0=eps0_tensor)
    res_sym = symmetry_u(u,v,0.0,domain,(ds(bottom_ID) + ds(left_ID)),h,eps0=eps0_tensor) 
    resI_u = interface_u(u,v,domain,dS(interface_ID),jump,w_0,h,C_u=10,eps0=eps0_tensor)

    res = res_u 
    res += resD_u_r 
    res += resD_u_t 
    res += res_sym
    if enrich:
        print('adding nitsches terms')
        res += resI_u


    J = ufl.derivative(res,u)
    res_form = fem.form(res)
    res_petsc = fem.petsc.assemble_vector(res_form)
    J_form = fem.form(J)
    J_petsc = fem.petsc.assemble_matrix(J_form)

    res_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    res_petsc.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    J_petsc.assemble()

    t_start = default_timer()
    EX_MAT_FILE = EX_MAT_DIR + subMeshType + "_mg.dat"
    readExOp=os.path.isfile(EX_MAT_FILE) 
    if readExOp and not genNew:
        print("reading in ExOp from file")
        M = PETSc.Mat().create(MPI.COMM_WORLD)
        viewer = PETSc.Viewer(MPI.COMM_WORLD).createBinary(EX_MAT_FILE,'r')
        M = (M.load(viewer))
        M.assemble()
    else:
        print('creating Ex Op')

        if enrich:
            print('enriching')
            M_u0 = common.interpolation_matrix_nonmatching_meshes(V_mg, V0)
            M_u1 = common.interpolation_matrix_nonmatching_meshes(V_mg, V1)
            new_mg_dofs_net = M_u0.shape[1] + new_mg_dofs
            M_lil_u0 =  sparse.lil_array(((M_u0.shape[0]),new_mg_dofs_net))
            M_lil_u1 =  sparse.lil_array(((M_u1.shape[0]),new_mg_dofs_net))
            M_lil_u0[:,np.arange(0,num_mg_dofs)] = M_u0
            M_lil_u1[:,np.arange(0,num_mg_dofs)] = M_u1
            
            for matID in range(mg_eBasisList.shape[1]):
                M_lil_u0[:,(mg_eBasisList[:,matID][mg_eBasisList[:,matID] >= 0 ])] = M_u0[:,(mg_eBasisList[:,matID] >= 0 )]*np.atleast_2d(enrichFuncs[matID].x.array).T
                M_lil_u1[:,(mg_eBasisList[:,matID][mg_eBasisList[:,matID] >= 0 ])] = M_u1[:,(mg_eBasisList[:,matID] >= 0 )]*np.atleast_2d(enrichFuncs[matID].x.array).T
            

            M_lil_multiField =  sparse.lil_array(((nFields*M_u0.shape[0]),nFields*new_mg_dofs_net))
            
            M_lil_multiField[V0_to_V, 0:new_mg_dofs_net] = M_lil_u0
            M_lil_multiField[V1_to_V,new_mg_dofs_net:(2*new_mg_dofs_net)] = M_lil_u0
            M_csr = M_lil_multiField.tocsr()

            M = PETSc.Mat().createAIJ(size=M_csr.shape,csr=(M_csr.indptr, M_csr.indices,M_csr.data))
            M.assemble()
            #M = common.createMEnrichVec(V,RKPMBasis,kernelFunc,newNp,eBasisList,enrichFuncs,nFields=2,)
        
        else:
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
        print('Saving fore-ground ExOp to file')
        viewer = PETSc.Viewer(MPI.COMM_WORLD).createBinary(EX_MAT_FILE, 'w')
        viewer(M)
        """
    t_stop = default_timer()
    t_ex = t_stop-t_start

    A,b = linAlgHelp.assembleLinearSystemBackground(J_petsc,-res_petsc,M)
    
    Vs += [V]
    As += [A]
    bs += [b]
    Ms += [M]
    mus += [mu]
    lams += [lam]
    domains +=[domain]
    u_exs +=[u_exact]
    ur_exs +=[ur_exact]
    us += [u]
    dxs += [dx] 
    t_exs += [t_ex]


A_tri,A_quad = As
b_tri,b_quad = bs
M_tri,M_quad = Ms
domain_tri,domain_quad = domains
u_ex_tri,u_ex_quad = u_exs 
u_tri,u_quad = us
dx_tri,dx_quad = dxs
t_ex_tri,t_ex_quad = t_exs

# add the two matrices
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

L2_error = fem.assemble_scalar(fem.form(ufl.inner(u_tri - u_ex_tri , u_tri  - u_ex_tri ) * dx_tri))
error_L2_tri  = np.sqrt(domain.comm.allreduce(L2_error, op=MPI.SUM))

H10_error = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(u_tri  - u_ex_tri ), ufl.grad(u_tri  - u_ex_tri )) *dx_tri ))
error_H10_tri  = np.sqrt(domain.comm.allreduce(H10_error, op=MPI.SUM))

L2 = fem.assemble_scalar(fem.form(ufl.inner(u_ex_tri, u_ex_tri) * dx_tri))
L2_tri  = np.sqrt(domain.comm.allreduce(L2, op=MPI.SUM))

H10= fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(u_ex_tri), ufl.grad(u_ex_tri)) *dx_tri ))
H10_tri  = np.sqrt(domain.comm.allreduce(H10, op=MPI.SUM))

L2_error = fem.assemble_scalar(fem.form(ufl.inner(u_quad - u_ex_quad , u_quad  - u_ex_quad ) * dx_quad))
error_L2_quad  = np.sqrt(domain.comm.allreduce(L2_error, op=MPI.SUM))

H10_error = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(u_quad  - u_ex_quad ), ufl.grad(u_quad  - u_ex_quad)) *dx_quad ))
error_H10_quad  = np.sqrt(domain.comm.allreduce(H10_error, op=MPI.SUM))

L2= fem.assemble_scalar(fem.form(ufl.inner(u_ex_quad , u_ex_quad ) * dx_quad))
L2_quad  = np.sqrt(domain.comm.allreduce(L2, op=MPI.SUM))

H10 = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(u_ex_quad ), ufl.grad(u_ex_quad)) *dx_quad ))
H10_quad  = np.sqrt(domain.comm.allreduce(H10, op=MPI.SUM))


tri_L2_norm = error_L2_tri/ L2_tri
tri_H10_norm = error_H10_tri/ H10_tri

quad_L2_norm = error_L2_quad/ L2_quad
quad_H10_norm = error_H10_quad/ H10_quad

net_L2_norm = (error_L2_quad + error_L2_tri) / (L2_quad + L2_tri)
net_H10_norm = (error_H10_quad + error_H10_tri) / (H10_quad + H10_tri)

print(f"Error_L2: {net_L2_norm}")
print(f"Error_H10: {net_H10_norm}")
print(f"Error_L2 (tris): {tri_L2_norm}")
print(f"Error_H10 (tris): {tri_H10_norm}")
print(f"Error_L2 (quads): {quad_L2_norm}")
print(f"Error_H10 (quads): {quad_H10_norm}")
print(f"Extraction Time (tris): {t_ex_tri}")
print(f"Extraction Time (quads): {t_ex_quad}")
print(f"Solver Time : {t_solve}")

#ref,n,L2,H1,kernel,supp,k,fg_r,t_ex_t,t_ex_q,t_solve
if write_file: 
    f = open(output_file,'a')
    f.write("\n")
    fs = str(actualNP)+","+opt+","+str(ref_val)+","+str(n)+","+str(net_L2_norm)+","+str(net_H10_norm)+","\
        + str(tri_L2_norm)+","+str(tri_H10_norm)+","\
        + str(quad_L2_norm)+","+str(quad_H10_norm)+","\
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
        u_exact = u_exs[i]
        ur_exact = ur_exs[i]
        domain = domains[i]
        mu = mus[i]
        lam = lams[i]

        folder = EX_MAT_DIR + "eigen2ex/"
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
        
        u0_expr_ex  = fem.Expression(u_exact[0] ,U0.element.interpolation_points())
        u0_plot_ex = fem.Function(U0)
        u0_plot_ex.interpolate(u0_expr_ex)
        u1_expr_ex  = fem.Expression(u_exact[1] ,U1.element.interpolation_points())
        u1_plot_ex = fem.Function(U1)
        u1_plot_ex.interpolate(u1_expr_ex)

        ur_expr_ex  = fem.Expression(ur_exact,U0.element.interpolation_points())
        ur_plot_ex = fem.Function(U0)
        ur_plot_ex.interpolate(ur_expr_ex)

        with io.VTXWriter(domain.comm, folder+"u0_ex_"+subMeshType+".bp", [u0_plot_ex], engine="BP4") as vtx:
            vtx.write(0.0)
        with io.VTXWriter(domain.comm, folder+"u1_ex_"+subMeshType+".bp", [u1_plot_ex], engine="BP4") as vtx:
            vtx.write(0.0)
        with io.VTXWriter(domain.comm, folder+"ur_ex_"+subMeshType+".bp", [ur_plot_ex], engine="BP4") as vtx:
            vtx.write(0.0)
        
        # plot strain 
        V_strain = fem.FunctionSpace(domain, ("DG", k-1))
        eps_soln = epsU(u)
        folder_ecomp = folder + "strain_components/"


        strainFileWriter(eps_soln[0,0],V_strain,folder_ecomp,"e00_"+subMeshType)
        strainFileWriter(eps_soln[1,0],V_strain,folder_ecomp,"e10_"+subMeshType)
        strainFileWriter(eps_soln[0,1],V_strain,folder_ecomp,"e01_"+subMeshType)
        strainFileWriter(eps_soln[1,1],V_strain,folder_ecomp,"e11_"+subMeshType)
        eps_sol_mag = ufl.sqrt(ufl.inner(eps_soln,eps_soln)) 
        strainFileWriter(eps_sol_mag,V_strain,folder,"eps_mag_"+subMeshType)

        eps_ex = epsU(u_ex)
        strainFileWriter(eps_ex[0,0],V_strain,folder_ecomp,"e00_ex_"+subMeshType)
        strainFileWriter(eps_ex[1,0],V_strain,folder_ecomp,"e10_ex_"+subMeshType)
        strainFileWriter(eps_ex[0,1],V_strain,folder_ecomp,"e01_ex_"+subMeshType)
        strainFileWriter(eps_ex[1,1],V_strain,folder_ecomp,"e11_ex_"+subMeshType)
        eps_ex_mag = ufl.sqrt(ufl.inner(eps_ex,eps_ex)) 
        strainFileWriter(eps_ex_mag,V_strain,folder,"eps_mag_ex_"+subMeshType)

        i += 1

