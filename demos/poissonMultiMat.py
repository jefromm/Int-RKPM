'''
A script to implement interpolation-based RKPM with FEniCSx 
Solving a multi material poisson problem on a 1x1 unit square 

Author: Jennifer E Fromm

'''

'''
import numpy as np 
import math
from InterpolationBasedRKPM import common, linAlgHelp
import os.path



from mpi4py import MPI
#note: need at least dolfinx version 0.5.0
import dolfinx
from dolfinx import mesh, fem, io, cpp
from dolfinx.fem import petsc
from dolfinx import cpp, la 
from dolfinx.cpp import io as c_io
import os
import ufl
from petsc4py import PETSc
from timeit import default_timer
from scipy import sparse, spatial
import itertools
'''


import numpy as np 
from IntRKPM import common, linAlgHelp, classicRKPMUtils
import os.path
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem
from dolfinx.fem import petsc
from scipy import sparse
import ufl

from timeit import default_timer

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
    

def interface_T(u,v,domain,dS,jump,weight,C_T=10, h=None):
    # specify normal directed away from inner section
    n = ufl.avg(weight*ufl.FacetNormal(domain))
    const = ufl.avg(jump*u) * ufl.dot(custom_avg((kappa*ufl.grad(v)),kappa,domain,h=h),n)*dS
    adjconst = ufl.avg(jump*v) * ufl.dot(custom_avg((kappa*ufl.grad(u)),kappa,domain,h=h),n)*dS
    gamma = gamma_int(C_T, kappa, domain,h=h)
    pen = gamma*ufl.avg(jump*u)*ufl.avg(jump*v)*dS
    return pen +const - adjconst


def dirichlet_T(T,q,Td,domain,ds,h, C_T=10):
    n = ufl.FacetNormal(domain)
    const = ((T-Td)*kappa*ufl.inner(ufl.grad(q), n))*ds
    adjconst = (q*kappa*ufl.inner(ufl.grad(T), n))*ds
    gamma = C_T *kappa / h 
    pen = (gamma*q*(T-Td))*ds
    return pen + const - adjconst


def Left(x):
    return np.isclose(x[0],0)
def Right(x):
    return np.isclose(x[0],L)
def Top(x):
    return np.isclose(x[1],L)
def Bottom(x):
    return np.isclose(x[1],0)

def Interface12(x):
    return np.isclose(x[0],L/5)
def Interface23(x):
    return np.isclose(x[0],4*L/5)


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
parser.add_argument('--fgs',dest='fgs',default=0,
                    help='Additional foreground refinement')
parser.add_argument('--supp',dest='supp',default=3,
                    help='Support size, default 3')
parser.add_argument('--kernel',dest='kernel',default='SPLIN3',
                    help='Kernel type, default is SPLIN3')
parser.add_argument('--nnn',dest='nnn',default=4,
                    help='Number of nearest neighbors')
parser.add_argument('--gt',dest='gT',default='jitter',
                    help="Grid type for RKPM points, either jitter (default) or random ")
parser.add_argument('--st',dest='st',default='cir',
                    help='support type, circular (cir), rectangular (rec), or tensor product (TP)')
parser.add_argument('--eps',dest='eps',default=0.5,
                    help="Float between 0 and 1 controlling level of perturbation, default 0.5")
parser.add_argument('--genNew',dest='genNew',default=False,
                    help='Generate new extraction operator, even if file exists')
parser.add_argument('--opt',dest='opt',default='_',
                    help='enrichment option to use- default is no enrichmentm ')
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
gridType = args.gT
supportType = args.st

n = int(args.n)
nnn = int(args.nnn)
k = args.k 
if k is None:
    k = n 
else:
    k = int(k)

ref_val = int(args.ref)
ref_scale = int(args.fgs)
supp = n + 1
gridEps = float(args.eps)
kernel = args.kernel
ref_dir = 'R' + str(ref_val) + '/'

EX_MAT_DIR= "n"+str(n)+"k"+str(k) + "fgs" + str(ref_scale)+ "/"
EX_MAT_FILE = opt 
EX_MAT_PATH = ref_dir + EX_MAT_DIR  +EX_MAT_FILE + ".dat"

folder = ref_dir + EX_MAT_DIR  +EX_MAT_FILE + "results/"

comm = MPI.COMM_WORLD
L = 1
aveKernel = L/ (5*2**ref_val)
def indicator0(coords):
    return (np.atleast_2d(np.less_equal(coords[:,0], L/5))).astype(int)
def indicator1(coords):
    return (np.atleast_2d(np.logical_and(np.greater(coords[:,0], L/5), np.less(coords[:,0], 4*L/5)))).astype(int)
def indicator2(coords):
    return (np.atleast_2d(np.greater_equal(coords[:,0], 4*L/5))).astype(int)




if opt == '_':
    print('Not using enrichment')
    enrich=False
elif opt == 'enrich':
    print( 'Using Heaviside enrichment')
    enrich = True
else:
    print("scaling option not supported")
    exit() 


# create RKPM basis 
RKPMBasis = classicRKPMUtils.RKPMBasisGenerator(ref_val)

t_file ="RKPMPoints/" + ref_dir + "tri.xdmf"

mesh_exists=os.path.isfile(t_file) 
if mesh_exists:
    RKPMBasis.readPoints(t_file)
else:
    print('warning: creating new pointset')
    RKPMBasis.makePoints(gridType=gridType, meshEps=gridEps, edgeNodes=[5,5])
    RKPMBasis.savePoints(t_file)

RKPMBasis.makeBasis(kernelType=kernel,polynomialOrder=n,supportSize=supp,numberNeighbors=nnn,supportType=supportType)

n_cells = (2**ref_scale)*(10)*(2**ref_val)
domain = mesh.create_rectangle(comm, [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
                               [n_cells, n_cells], mesh.CellType.quadrilateral)

def Omega0(x):
    return x[0] <= L/5 
def Omega1(x):
    return np.logical_and(x[0] >= L/5, x[0] <= 4*L/5)
def Omega2(x):
    return x[0] >= 4*L/5 

cells_0 = mesh.locate_entities(domain, domain.topology.dim, Omega0)
cells_1 = mesh.locate_entities(domain, domain.topology.dim, Omega1)
cells_2 = mesh.locate_entities(domain, domain.topology.dim, Omega2)

V = fem.FunctionSpace(domain,('DG',(k)))
Q = fem.FunctionSpace(domain, ("DG", 0))

if opt == 'enrich':
    #identify nodes in range, and duplicate 
    omega0Q = fem.Function(Q)
    omega0Q.x.array[cells_0] = np.ones_like(cells_0)*1.0
    omega0Q.x.array[cells_1] = np.ones_like(cells_1)*0.0
    omega0Q.x.array[cells_2] = np.ones_like(cells_2)*0.0
    omega0Q.x.scatter_forward()
    omega1Q = fem.Function(Q)
    omega1Q.x.array[cells_0] = np.ones_like(cells_0)*0.0
    omega1Q.x.array[cells_1] = np.ones_like(cells_1)*1.0
    omega1Q.x.array[cells_2] = np.ones_like(cells_2)*0.0
    omega1Q.x.scatter_forward()

    omega2Q = fem.Function(Q)
    omega2Q.x.array[cells_0] = np.ones_like(cells_0)*0.0
    omega2Q.x.array[cells_1] = np.ones_like(cells_1)*0.0
    omega2Q.x.array[cells_2] = np.ones_like(cells_2)*1.0
    omega2Q.x.scatter_forward()

    omega0V = fem.Function(V)
    omega0V.interpolate(omega0Q)
    omega1V = fem.Function(V)
    omega1V.interpolate(omega1Q)
    omega2V = fem.Function(V)
    omega2V.interpolate(omega2Q)

    
    kernelRads = RKPMBasis.domainKernel
    numNodes = len(kernelRads)
    # 2 interfaces, one function per interface 
    eBasisList = np.ones((numNodes, 3))*-1
    #enrichFuncs = [H12_l, H12_r, H23_l, H23_r]
    enrichFuncs = [omega0V,omega1V,omega2V]
    nextNodeID = numNodes 
    i = 0 
    for node in RKPMBasis.nodeCoords:
        interface = False 
        dist12 = abs(node[0] - L/5)
        dist23 = abs(node[0] - 4*L/5)
        if dist12 <= kernelRads[i]:
            interface = True
            eBasisList[i,0] = i
            eBasisList[i,1] = nextNodeID 
            nextNodeID += 1 
        if dist23 <= kernelRads[i]:
            if not interface:
                # node is not at all in domain 1 
                eBasisList[i,1] = i
            eBasisList[i,2] = nextNodeID
            nextNodeID += 1
        i+= 1
    newNodeNum = nextNodeID 
    

    
kappa = fem.Function(Q)
kappa.x.array[cells_0] = np.ones_like(cells_0)*1.0
kappa.x.array[cells_1] = np.ones_like(cells_1)*0.5
kappa.x.array[cells_2] = np.ones_like(cells_2)*1
kappa.x.scatter_forward()

const = fem.Function(Q)
const.x.array[cells_0] = np.ones_like(cells_0)*0.0
const.x.array[cells_1] = np.ones_like(cells_1)*(L/10)
const.x.array[cells_2] = np.ones_like(cells_2)*(-3*L/10)
const.x.scatter_forward()

weight = fem.Function(Q)
weight.x.array[cells_0] = np.ones_like(cells_0)*0.0
weight.x.array[cells_1] = np.ones_like(cells_1)*2.0
weight.x.array[cells_2] = np.ones_like(cells_2)*0.0
weight.x.scatter_forward()

jump = fem.Function(Q)
jump.x.array[cells_0] = np.ones_like(cells_0)*-2.0
jump.x.array[cells_1] = np.ones_like(cells_1)*2.0
jump.x.array[cells_2] = np.ones_like(cells_2)*-2.0
jump.x.scatter_forward()


# facet markers, user specified
dim =2
left_ID = 1
right_ID = 2
interface_12_ID = 3
interface_23_ID = 4
top_ID = 5
bottom_ID = 6
facet_markers = [left_ID, right_ID, interface_12_ID,interface_23_ID,top_ID, bottom_ID]
facet_functions = [Left, Right, Interface12,Interface23,Top,Bottom]
num_facet_phases =len(facet_markers)
domain.topology.create_connectivity(dim-1, dim)
num_facets = domain.topology.index_map(dim-1).size_global
f_to_c_conn = domain.topology.connectivity(dim-1,dim)
facets = np.asarray([],dtype=np.int32)
facets_mark = np.asarray([],dtype=np.int32)
for phase in range(num_facet_phases):
    facets_phase = mesh.locate_entities(domain, dim-1, facet_functions[phase])
    facets_phase_mark = np.full_like(facets_phase, facet_markers[phase])
    facets= np.hstack((facets,facets_phase))
    facets_mark = np.hstack((facets_mark,facets_phase_mark))
sorted_facets = np.argsort(facets)
ft = mesh.meshtags(domain,dim-1,facets[sorted_facets], facets_mark[sorted_facets])

dx_custom = ufl.Measure('dx',domain=domain, metadata={'quadrature_degree': 2*k})
ds = ufl.Measure("ds",domain=domain,subdomain_data=ft,metadata={'quadrature_degree': 4*k})
dS = ufl.Measure("dS",domain=domain,subdomain_data=ft,metadata={'quadrature_degree': 4*k})

x = ufl.SpatialCoordinate(domain)

V_flux =  fem.FunctionSpace(domain, ("DG", k-1))
u = fem.Function(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)


def T_source_ufl(x): 
    x_bar = x[0] - (L/5)
    return ufl.sin(5*ufl.pi*x[1]/(3*L))*ufl.sin(5*ufl.pi*x_bar/(3*L))/kappa

u_ex=T_source_ufl(x)
f = -ufl.div(ufl.grad(u_ex))*kappa 

h = 1/n_cells

res_T = kappa* ufl.inner(ufl.grad(u),ufl.grad(v))*(dx_custom) - ufl.inner(v,f)*(dx_custom)
resD_T_l = dirichlet_T(u,v,u_ex,domain,ds(left_ID),h)
resD_T_r = dirichlet_T(u,v,u_ex,domain,ds(right_ID),h)
resD_T_t = dirichlet_T(u,v,u_ex,domain,ds(top_ID),h)
resD_T_b = dirichlet_T(u,v,u_ex,domain,ds(bottom_ID),h)
resI_12 = interface_T(u,v,domain,dS(interface_12_ID),jump,weight,C_T=10,h=h)
resI_23 = interface_T(u,v,domain,dS(interface_23_ID),jump,weight,C_T=10,h=h)

res = res_T 
res += resD_T_l 
res += resD_T_r 
res += resD_T_t 
res += resD_T_b
if enrich:
    print('adding nitsches terms')
    res += resI_12 
    res += resI_23


J = ufl.derivative(res,u)
res_form = fem.form(res)
res_petsc = fem.petsc.assemble_vector(res_form)
J_form = fem.form(J)
J_petsc = fem.petsc.assemble_matrix(J_form)

J_petsc.assemble()

t_start = default_timer()
readExOp=os.path.isfile(EX_MAT_PATH) 
if readExOp and not genNew:
    print("reading in ExOp from file")
    M = PETSc.Mat().create(MPI.COMM_WORLD)
    viewer = PETSc.Viewer(MPI.COMM_WORLD).createBinary(EX_MAT_PATH,'r')
    M = (M.load(viewer))
    M.assemble()
else:
    print('creating Ex Op')
    if enrich:
        print('enriching')
        newNp = newNodeNum
        M_preEnrich = common.createM(V,RKPMBasis,returnAsSparse=True)

        M_lil =  sparse.lil_array(((M_preEnrich.shape[0]),newNp))
        #first set M_lil to the original list, which will take care of any non enriched functions 
        M_lil[:,np.arange(0,RKPMBasis.nP)] = M_preEnrich
        # then replace all enriched shape functions with their proper values
        for matID in range(eBasisList.shape[1]):
            M_lil[:,(eBasisList[:,matID][eBasisList[:,matID] >= 0 ])] = M_preEnrich[:,(eBasisList[:,matID] >= 0 )]*np.atleast_2d(enrichFuncs[matID].x.array).T
        M_csr = M_lil.tocsr()
        M = PETSc.Mat().createAIJ(size=M_csr.shape,csr=(M_csr.indptr, M_csr.indices,M_csr.data))
        M.assemble()
    else:
        M = common.createM(V,RKPMBasis)
    
    """
    Note: it is highly recomended to set up the directory structure to save and reuse extraction operators
    print('Saving ExOp to file')
    viewer = PETSc.Viewer(MPI.COMM_WORLD).createBinary(EX_MAT_PATH, 'w')
    viewer(M)
    """

t_stop = default_timer()
t_ex = t_stop-t_start

A,b = linAlgHelp.assembleLinearSystemBackground(J_petsc,-res_petsc,M)
x = A.createVecLeft()

t_start = default_timer()
linAlgHelp.solveKSP(A,b,x,monitor=False,method='mumps')
t_stop = default_timer()
t_solve = t_stop-t_start

common.transferToForeground(u, x, M)


T_sum = fem.assemble_scalar(fem.form(ufl.inner(u , u) * dx_custom))
T_sum_assembled  = domain.comm.allreduce(T_sum, op=MPI.SUM)

L2 = fem.assemble_scalar(fem.form(ufl.inner(u-u_ex, u-u_ex) * dx_custom))
L2_assembled  = domain.comm.allreduce(L2, op=MPI.SUM)

H1 = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(u-u_ex), ufl.grad(u-u_ex)) * dx_custom))
H1_assembled  = domain.comm.allreduce(H1, op=MPI.SUM)

L2_norm = fem.assemble_scalar(fem.form(ufl.inner(u_ex, u_ex) * dx_custom))
L2_assembled_norm  = domain.comm.allreduce(L2_norm, op=MPI.SUM)

H1_norm = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(u_ex), ufl.grad(u_ex)) * dx_custom))
H1_assembled_norm  = domain.comm.allreduce(H1_norm, op=MPI.SUM)


L2_error = np.sqrt(L2_assembled / L2_assembled_norm)
H10_error = np.sqrt(H1_assembled / H1_assembled_norm)
sum_T = np.sqrt(T_sum_assembled)

print(f"Error L2: {L2_error}")
print(f"Error H10: {H10_error}")
print(f"Sum T: {sum_T}")
print(f"Extraction Time: {t_ex}")
print(f"Solver Time : {t_solve}")

#ref,n,L2,H1,kernel,supp,k,fg_r,t_ex_t,t_ex_q,t_solve
if write_file: 
    f = open(output_file,'a')
    f.write("\n")
    fs = EX_MAT_FILE + "," + str(ref_val)+","+str(n)+","+str(L2_error)+","+str(H10_error)+","\
        + kernel+","+str(supp) +","+ str(k)+","+ str(ref_scale)+","+ \
            str(t_ex)+","+str(t_solve)
    f.write(fs)
    f.close()


if visOutput:
    if k > 1:
        fluxFileWriter = common.outputVTX
    else: 
        fluxFileWriter = common.outputXDMF
    common.outputVTX(u,V,folder,'u')
    common.outputVTX(u_ex,V,folder,'u_ex')
    q_flux = ufl.grad(u)
    fluxFileWriter(q_flux[0],V_flux,folder,"q0")
    fluxFileWriter(q_flux[1],V_flux,folder,"q1")
    q_ex_flux = ufl.grad(u_ex)
    fluxFileWriter(q_ex_flux[0],V_flux,folder,"q0_ex")
    fluxFileWriter(q_ex_flux[1],V_flux,folder,"q1_ex")



    # from https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html#making-curve-plots-throughout-the-domain
    # Plot u along a line y = const and compare with exact solution
    tol = 0.001  # Avoid hitting the outside of the domain
    pltRes = 100*(2**ref_val) + 1 
    x = np.linspace(0.0 + tol, 1.0 - tol, pltRes)
    y = np.ones_like(x)*0.3
    points = np.zeros((3, pltRes))
    points[0] = x
    points[1] = y
    u_values = []
    dudx0_values = []
    dudx1_values = []

    q_flux = ufl.grad(u)
    dudx0_expr = fem.Expression(q_flux[0], V_flux.element.interpolation_points())
    dudx0= fem.Function(V_flux)
    dudx0.interpolate(dudx0_expr)
    dudx1_expr = fem.Expression(q_flux[1], V_flux.element.interpolation_points())
    dudx1= fem.Function(V_flux)
    dudx1.interpolate(dudx1_expr)


    ex_u_values = []
    ex_dudx0_values = []
    ex_dudx1_values = []

    ex_V=  fem.FunctionSpace(domain, ("DG", k+6))
    ex_u_expr = fem.Expression(u_ex, ex_V.element.interpolation_points())
    ex_u= fem.Function(ex_V)
    ex_u.interpolate(ex_u_expr)

    ex_V_flux =  fem.FunctionSpace(domain, ("DG", k+5))
    ex_q_flux = ufl.grad(u_ex)
    ex_dudx0_expr = fem.Expression(ex_q_flux[0], ex_V_flux.element.interpolation_points())
    ex_dudx0= fem.Function(ex_V_flux)
    ex_dudx0.interpolate(ex_dudx0_expr)
    ex_dudx1_expr = fem.Expression(ex_q_flux[1], ex_V_flux.element.interpolation_points())
    ex_dudx1= fem.Function(ex_V_flux)
    ex_dudx1.interpolate(ex_dudx1_expr)



    from dolfinx import geometry
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = u.eval(points_on_proc, cells)
    dudx0_values = dudx0.eval(points_on_proc, cells)
    dudx1_values = dudx1.eval(points_on_proc, cells)
    ex_u_values = ex_u.eval(points_on_proc, cells)
    ex_dudx0_values = ex_dudx0.eval(points_on_proc, cells)
    ex_dudx1_values = ex_dudx1.eval(points_on_proc, cells)

    import pandas as pd
    # save data as csv w/ dataframe
    data = {'x0': points_on_proc[:, 0], 'x1': points_on_proc[:, 1], 'T_ex': ex_u_values[:,0], 'dx0_ex': ex_dudx0_values[:,0],'dx1_ex': ex_dudx1_values[:,0],
            'T': u_values[:,0], 'dx0': dudx0_values[:,0], 'dx1': dudx1_values[:,0]}

    df = pd.DataFrame(data)
    df_file = 'matchingMidlinePlots/' + opt + '/R' +str(ref_val)+"n"+str(n) + 'Int.csv' 
    df.to_csv(df_file, index=False)


    from matplotlib import pyplot as plt
    fig, axes= plt.subplots(3, 1, figsize=(12, 8))

    axes[0].plot(points_on_proc[:, 0],ex_u_values[:,0], label = 'exact', linestyle= 'dotted')
    axes[0].plot(points_on_proc[:, 0],u_values[:,0], label = 'solution')

    axes[1].plot(points_on_proc[:, 0],ex_dudx0_values[:,0], label = 'exact', linestyle= 'dotted')
    axes[1].plot(points_on_proc[:, 0],dudx0_values[:,0], label = 'solution')

    axes[2].plot(points_on_proc[:, 0],ex_dudx1_values[:,0], label = 'exact', linestyle= 'dotted')
    axes[2].plot(points_on_proc[:, 0],dudx1_values[:,0], label = 'solution')

    y_labels = ['solution', 'dx', 'dy'] 
    axe_count = 0
    for ax in axes:
        ax.legend( fontsize=12,frameon=False)#, loc='lower right')
        ax.set_xlabel('x', fontsize='x-large')
        ax.set_ylabel(y_labels[axe_count], fontsize='x-large')
        #ax.set_title(axis_titles[axe_count],fontsize='xx-large')
        axe_count += 1


    fig.suptitle('Solution along Y=0.3',fontsize='xx-large')
    plt.tight_layout()
    fig_file = 'matchingMidlinePlots/'+opt+'/R'+str(ref_val)+'n'+str(n)+'k'+str(k)+'fgs'+str(ref_scale)+'.png'
    plt.savefig(fig_file)
