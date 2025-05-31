'''
A script to implement interpolation-based RKPM with FEniCSx 
Solving the poisson problem on a 1x1 unit square 

Author: Jennifer E Fromm

'''

import numpy as np 
from IntRKPM import common, linAlgHelp, classicRKPMUtils
import os.path
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl

from timeit import default_timer

def lap(x):
    return ufl.div(ufl.grad(x))

def dirichlet_p(T,q,Td,domain,ds,h, C_T=10):
    n = ufl.FacetNormal(domain)
    const = ((T-Td)*ufl.inner(ufl.grad(q), n))*ds
    adjconst = (q*ufl.inner(ufl.grad(T), n))*ds
    gamma = C_T / h 
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


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--wf',dest='wf',default=False,
                    help='Write error to file (True/False)')
parser.add_argument('--vd',dest='vd',default=False,
                    help='visualize data by generating mesh files with solution, defualt False')
parser.add_argument('--of',dest='of',default="poissonIntData.csv",
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
EX_MAT_FILE = 'ex'
EX_MAT_PATH = "ExOps/" + ref_dir + EX_MAT_DIR  +EX_MAT_FILE + ".dat"

comm = MPI.COMM_WORLD
L = 1
aveKernel = L/ (5*2**ref_val)

RKPMBasis = classicRKPMUtils.RKPMBasisGenerator(ref_val)

t_file = "RKPMPoints/" + ref_dir + "tri.xdmf"

mesh_exists=os.path.isfile(t_file) 
if mesh_exists:
    RKPMBasis.readPoints(t_file)
else:
    print('warning: creating new pointset')
    RKPMBasis.makePoints(gridType=gridType, meshEps=gridEps, edgeNodes=[5,5])
    RKPMBasis.savePoints(t_file)

RKPMBasis.makeBasis(kernelType=kernel,polynomialOrder=n,supportSize=supp,numberNeighbors=nnn,supportType=supportType)

n_cells = (2**ref_scale)*(5)*(2**ref_val)
domain = mesh.create_rectangle(comm, [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
                               [n_cells, n_cells], mesh.CellType.quadrilateral)

V = fem.FunctionSpace(domain,('DG',(k)))

# facet markers, user specified
dim =2
left_ID = 1
right_ID = 2
top_ID = 3
bottom_ID = 4
facet_markers = [left_ID, right_ID,top_ID, bottom_ID]
facet_functions = [Left, Right,Top,Bottom]
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
x = ufl.SpatialCoordinate(domain)

V_flux =  fem.FunctionSpace(domain, ("DG", k-1))


u = fem.Function(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)

a = 5
def T_source_ufl(x): 
    return ufl.sin(a*x[0] + 0.1)*ufl.sin(a*x[1] + 0.1)

u_ex=T_source_ufl(x)
h = 1/n_cells


f = -lap(u_ex)
res_T =  ufl.inner(ufl.grad(u),ufl.grad(v))*(dx_custom) - ufl.inner(v,f)*(dx_custom)
resD_T_l = dirichlet_p(u,v,u_ex,domain,ds(left_ID),h)
resD_T_r = dirichlet_p(u,v,u_ex,domain,ds(right_ID),h)
resD_T_t = dirichlet_p(u,v,u_ex,domain,ds(top_ID),h)
resD_T_b = dirichlet_p(u,v,u_ex,domain,ds(bottom_ID),h)

res = res_T 
res += resD_T_l 
res += resD_T_r 
res += resD_T_t 
res += resD_T_b

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
    M = common.createM(V,RKPMBasis)
    """
    Note: it is highly recomended to set up the directory structure to save and reuse extraction operators
    print('Saving ExOp to file')
    viewer = PETSc.Viewer(MPI.COMM_WORLD).createBinary(EX_MAT_PATH, 'w')
    viewer(M)
    """

t_stop = default_timer()
t_ex = t_stop-t_start
sizes= M.getSizes()
bg_dofs = sizes[1][0]
fg_dofs = sizes[0][0]

'''
#debugging option to verify nitsches method
print("using native FEM space, no extraction-")
# note- need to go back and use a CG space instead of DG, 
#       and swap transferToForeground for .scatter_forward after solve
A = J_petsc
b = -res_petsc 
x = la.create_petsc_vector_wrap(u.x)
'''


A,b = linAlgHelp.assembleLinearSystemBackground(J_petsc,-res_petsc,M)
x = A.createVecLeft()

t_start = default_timer()
linAlgHelp.solveKSP(A,b,x,monitor=False,method='mumps')


t_stop = default_timer()
t_solve = t_stop-t_start

#comment out if using native FEM space
common.transferToForeground(u, x, M)

#uncomment if using native FEM space
#u.x.scatter_forward()


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

if write_file: 
    f = open(output_file,'a')
    f.write("\n")
    fs = str(bg_dofs)+","+str(fg_dofs)+ "," + str(ref_val)+","+str(n)+","\
        + str(L2_error)+","+str(H10_error)+","\
        + kernel+","+str(supp) +","+ str(k)+","+ str(ref_scale)+"," \
        + str(t_ex)+","+str(t_solve)
    f.write(fs)
    f.close()


if k > 1:
    fluxFileWriter = common.outputVTX
else: 
    fluxFileWriter = common.outputXDMF

folder = "poissonData/"
if visOutput:
    common.outputVTX(u,V,folder,'u')
    common.outputVTX(u_ex,V,folder,'u_ex')
    q_flux = ufl.grad(u)
    fluxFileWriter(q_flux[0],V_flux,folder,"q0")
    fluxFileWriter(q_flux[1],V_flux,folder,"q1")
    q_ex_flux = ufl.grad(u_ex)
    fluxFileWriter(q_ex_flux[0],V_flux,folder,"q0_ex")
    fluxFileWriter(q_ex_flux[1],V_flux,folder,"q1_ex")       
