'''
A script to solve the poisson problem on a 1x1 unit square with RKPM, using guass quadrature 

Author: Jennifer E Fromm

'''

import numpy as np 
from IntRKPM import linAlgHelp, classicRKPMUtils
import os.path
import matplotlib.pyplot as plt 
import os
from timeit import default_timer
from scipy import sparse

 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--wf',dest='wf',default=False,
                    help='Write error to file (True/False)')
parser.add_argument('--vd',dest='vd',default=False,
                    help='visualize data by generating plots with solution, default False')
parser.add_argument('--of',dest='of',default="poissonData.csv",
                    help='Output file destination')
parser.add_argument('--n',dest='n',default=1,
                    help='RK polynomial order')
parser.add_argument('--ref',dest='ref',default=0,
                    help='Refinement level')
parser.add_argument('--supp',dest='supp',default=None,
                    help='Support size, default n + 1')
parser.add_argument('--kernel',dest='kernel',default='SPLIN3',
                    help='Kernel type, default is SPLIN3')
parser.add_argument('--nnn',dest='nnn',default=4,
                    help='Number of nearest neighbors')
parser.add_argument('--gt',dest='gT',default='jitter',
                    help="Grid type for RKPM points, either jitter (default) or random ")
parser.add_argument('--eps',dest='eps',default='0.5',
                    help="Randomness level, float between 0.0 and 1.0 (0== uniform grid)")
parser.add_argument('--st',dest='st',default='cir',
                    help='support type, circular (cir), rectangular (rec), or tensor product (TP)')
parser.add_argument('--gp',dest='gp',default=5,
                    help='Number of Gauss points per integration cell (default = 5)')
parser.add_argument('--gpE',dest='gpE',default=None,
                    help='Number of Gauss points per error integration cell (default is same as method gp)')
parser.add_argument('--gcRef',dest='gcRef',default=0,
                    help='Refinement level of gauss integration cell relative to nodal spacing (default = 0)')
parser.add_argument('--gcRefE',dest='gcRefE',default=None,
                    help='Refinement level of gauss error integration cell relative to nodal spacing (default is same as method gcRef)')
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

output_file = args.of
gridType = args.gT
supportType = args.st

n = int(args.n)
nnn = int(args.nnn)
ref_val = int(args.ref)
supp = args.supp
if supp == None:
    supp = n + 1
else:
    supp = float(supp)
kernel = args.kernel
eps_val  = float(args.eps)
gp = int(args.gp)
gcRef = int(args.gcRef)

gpE = args.gpE
if gpE is None:
    gpE = gp
else:
    gpE = int(gpE)
gcRefE = args.gcRefE
if gcRefE is None:
    gcRefE = gcRef
else:
    gcRefE = int(gcRefE)

if (gp == gpE) and (gcRef == gcRefE): 
    reuseGP = True
else:
    reuseGP = False

ref_dir = 'R' + str(ref_val) + '/'

corner0 = [0.0,0.0]
corner1 = [1.0,1.0]
L = 1
aveKernel = L/ (5*2**ref_val)

RKPMBasis = classicRKPMUtils.RKPMBasisGenerator(ref_val)
def norm(coord0,coord1):
    return np.sqrt((coord0[0] - coord1[0])**2 + (coord0[1] - coord1[1])**2)

t_file = "RKPMPoints/" + ref_dir + "tri.xdmf"

mesh_exists=os.path.isfile(t_file) 
if mesh_exists:
    RKPMBasis.readPoints(t_file)
else:
    print("warning: generating new point set")
    RKPMBasis.makePoints(gridType="jitter", meshEps=eps_val,edgeNodes=5)
    RKPMBasis.savePoints(t_file)
RKPMBasis.makeBasis(kernelType=kernel,polynomialOrder=n,supportSize=supp,numberNeighbors=nnn,supportType=supportType)

# define geometry - one by one square w/ square integration cells 

# enforce dirichlet BC on each edges w/ Nitsches 

############# Compute quadrature points   ##################
print('Computing quad points')

nGP =gp
nGPCell = nGP*nGP
gcPerEdgeC = 5* (2**gcRef)
gcPerEdge = gcPerEdgeC*(2**ref_val)
nGC = gcPerEdge*gcPerEdge
nGPBulk = nGPCell*nGC
gcSize0 = (corner1[0]- corner0[0])/ gcPerEdge
gcSize1 = (corner1[1]- corner0[1])/ gcPerEdge

gcVerts0 = np.linspace(corner0[0],corner1[0], gcPerEdge+1)
gcVerts1 = np.linspace(corner0[1],corner1[1], gcPerEdge+1)

gpX0 = []
gpX1 = []
gpW = []
for ID0 in range(gcPerEdge):
    for ID1 in range(gcPerEdge):
        X0, W0 = classicRKPMUtils.getGaussQuad(nGP, gcVerts0[ID0], gcVerts0[(ID0+1)])
        X1, W1 = classicRKPMUtils.getGaussQuad(nGP, gcVerts1[ID1], gcVerts1[(ID1+1)])
        X0cell,X1cell = np.meshgrid(X0, X1)
        W0cell,W1cell  = np.meshgrid(W0, W1) 
        Wcell = W0cell*W1cell
        gpX0 = np.hstack((gpX0,X0cell.flatten()))
        gpX1 = np.hstack((gpX1,X1cell.flatten()))
        gpW = np.hstack((gpW,Wcell.flatten()))

# create boundary nodes, stack them in order
#gamma0 - left
gamma0 = np.vstack((np.ones_like(gcVerts1[1:])*corner0[0],gcVerts1[1:])).T
#gamma1 - top
gamma1 = np.vstack((gcVerts0[1:],np.ones_like(gcVerts0[1:])*corner1[1])).T
#gamma2 - right
gamma2 = np.vstack((np.ones_like(gcVerts1[1:])*corner1[0],np.flip(gcVerts1)[1:])).T
#gamma3 - bottom
gamma3 = np.vstack((np.flip(gcVerts0)[1:],np.ones_like(gcVerts0[1:])*corner0[1])).T

rotateMat = np.asarray([[np.cos(np.pi/2), -np.sin(np.pi/2)],[np.sin(np.pi/2), np.cos(np.pi/2)]] )

gammaGCverts = np.vstack((gamma0,gamma1,gamma2,gamma3))
BCgpX0 = []
BCgpX1 = []
BCgpW = []
BCgpNormals = None
for i in range(gammaGCverts.shape[0]):
    if i == gammaGCverts.shape[0]-1:
        coord0 = gammaGCverts[i,:]
        coord1 = gammaGCverts[0,:]
    else:
        coord0 = gammaGCverts[i,:]
        coord1 = gammaGCverts[i+1,:]
    X0,_ = classicRKPMUtils.getGaussQuad(nGP, coord0[0], coord1[0])
    X1,_ = classicRKPMUtils.getGaussQuad(nGP, coord0[1], coord1[1])
    L_seg = norm(coord0,coord1)
    _,W = classicRKPMUtils.getGaussQuad(nGP, 0.0, L_seg)
    X10 = coord1 - coord0 
    X10_normal = np.matmul(X10,rotateMat)
    X10_normal = X10_normal / (norm(X10_normal, [0,0]))
    gpNormals = np.tile(X10_normal,(nGP,1))

    BCgpX0 = np.hstack((BCgpX0,X0))
    BCgpX1 = np.hstack((BCgpX1,X1))
    BCgpW = np.hstack((BCgpW,W))
    if BCgpNormals is not None:
        BCgpNormals= np.vstack((BCgpNormals,gpNormals))
    else:
        BCgpNormals = gpNormals 

nGPBoundary = BCgpW.size

############# End compute quadrature points   ##################


############# Compute SHP at quadrature points   ##################
print('Computing shape functions at quad points')
t_start = default_timer()

eval_pts = np.vstack((gpX0, gpX1)).T
SHP, SHPdX0, SHPdX1 = RKPMBasis.evalSHP(eval_pts,returnFirstDerivative=True)

BCeval_pts = np.vstack((BCgpX0, BCgpX1)).T
BC_SHP, BC_SHPdX0, BC_SHPdX1 = RKPMBasis.evalSHP(BCeval_pts,returnFirstDerivative=True)

t_stop = default_timer()
t_SHP_comp = t_stop - t_start

t_start = default_timer()


############# End compute SHP at quadrature points   ##################

############# Assemble matrices  ##################
print('Assembling matrices')    
# solving simple poisson's problem Kx=b
domainKernel = RKPMBasis.domainKernel

beta_param = 10
beta = beta_param/ domainKernel # 
sgn = -1.0 # for nonsymmetric 

a = 5 # wave length parameter
def T_ex(x):
    return np.sin(a*x[0] + 0.1)*np.sin(a*x[1] + 0.1)

def gradT_ex(x): 
    dx0 = a*np.cos(a*x[0] + 0.1)*np.sin(a*x[1] + 0.1) 
    dx1 = a*np.sin(a*x[0] + 0.1)*np.cos(a*x[1] + 0.1)
    return np.vstack((dx0,dx1))

def f_b(x):
    return 2*a*a*np.sin(a*x[0] + 0.1)*np.sin(a*x[1] + 0.1)

#bulk
# KIJ = : grad(T)* grad(theta)* kappa
#   B'B kappa
# F_b = theta* f_b(qpt)
#   Psi' f_b(qpt)
t_start = default_timer()

nP = RKPMBasis.nP
coords = np.vstack((gpX0,gpX1))
f_b_v = f_b(coords)
B_all =sparse.lil_array((2*nGPBulk,nP))
B_all[::2] = SHPdX0
B_all[1::2] = SHPdX1 
W_2 =np.zeros((2*nGPBulk,1))
W_2[::2] = np.asarray([gpW]).T
W_2[1::2] = np.asarray([gpW]).T


KIJ = (B_all).T@ (W_2*B_all)
F = (gpW.T*f_b_v)@SHP

#boundary- nitches method
# KIJ_n = kappa* (T * grad(theta)*n + theta*grad(T)*n) - beta*theta*T)
#   kappa( Psi' * B*n + (B*n)' *Psi ) - beta Psi' Psi 

# F_n = -kappa*T_ex*grad(theta)*n - beta*theta*T_ex 
#   - kappa*T_ex(qpt)*(B*n)' - beta*T_ex(qpt)*Psi'
BC_B_all =sparse.lil_array((2*nGPBoundary,nP))
BC_B_all[::2] = BC_SHPdX0
BC_B_all[1::2] = BC_SHPdX1 

norm_mat1 = np.diagflat(BCgpNormals[:,0])
norm_mat2 = np.diagflat(BCgpNormals[:,1])
norm_mat = np.zeros((nGPBoundary,2*nGPBoundary))
norm_mat[:,::2] = norm_mat1
norm_mat[:,1::2] = norm_mat2

BCcoords = np.vstack((BCgpX0,BCgpX1))
BCT_ex_v= T_ex(BCcoords)


F_pen_v = (BCgpW.T*BCT_ex_v)@BC_SHP* beta
F_const_v = sgn*(BCgpW.T*BCT_ex_v)@(norm_mat@BC_B_all)
F += F_pen_v + F_const_v

K_pen_v = (beta*BC_SHP).T@((np.asarray([BCgpW]).T*BC_SHP))
K_const_v = ((np.asarray([BCgpW]).T*BC_SHP)).T@(norm_mat@BC_B_all)
K_Aconst_v = sgn* K_const_v.T

KIJ += K_pen_v + K_const_v + K_Aconst_v

t_stop = default_timer()
t_assemble = t_stop - t_start


############# End assemble matrices  ##################
    
############# Solve linear system  ##################    
print('Solving linear system ')

KIJ_petsc = linAlgHelp.np2PMat(KIJ)
F_petsc = linAlgHelp.np2PVec(F)


x = KIJ_petsc.createVecLeft()


t_start = default_timer()

linAlgHelp.solveKSP(KIJ_petsc,F_petsc,x,method='mumps')

t_stop = default_timer()
t_solve = t_stop-t_start

nodal_soln = np.asarray([linAlgHelp.p2npVec(x)])

############# End solve linear system  ##################   


############# Compute error calc quadrature points   ##################
print('Computing quad points for error calculation')

reuseGP = True
if reuseGP:
    nGPE =gpE
    gcPerEdgeCE=5* (2**gcRef)
    gpX0E = gpX0
    gpX1E = gpX1
    gpWE = gpW
    nGPBulkE = nGPBulk 
    
else:
    nGPE =gpE
    nGPCellE = nGPE*nGPE
    gcPerEdgeCE=5* (2**gcRef)
    gcPerEdgeE = gcPerEdgeCE*(2**ref_val)
    nGCE = gcPerEdgeE*gcPerEdgeE
    nGPBulkE = nGPCellE*nGCE

    gcVerts0E = np.linspace(corner0[0],corner1[0], gcPerEdgeE+1)
    gcVerts1E = np.linspace(corner0[1],corner1[1], gcPerEdgeE+1)

    gpX0E = []
    gpX1E = []
    gpWE = []
    for ID0 in range(gcPerEdgeE):
        for ID1 in range(gcPerEdgeE):
            X0, W0 = classicRKPMUtils.getGaussQuad(nGPE, gcVerts0E[ID0], gcVerts0E[(ID0+1)])
            X1, W1 = classicRKPMUtils.getGaussQuad(nGPE, gcVerts1E[ID1], gcVerts1E[(ID1+1)])
            X0cell,X1cell = np.meshgrid(X0, X1)
            W0cell,W1cell  = np.meshgrid(W0, W1) 
            Wcell = W0cell*W1cell
            gpX0E = np.hstack((gpX0E,X0cell.flatten()))
            gpX1E = np.hstack((gpX1E,X1cell.flatten()))
            gpWE = np.hstack((gpWE,Wcell.flatten()))
    W_2E =np.zeros((2*nGPE,1))
    W_2E[::2] = np.asarray([gpWE]).T
    W_2E[1::2] = np.asarray([gpWE]).T


############# End compute error calc quadrature points   ##################

############# Compute SHP functions at error calc quadrature points ##################
print('Computing shape functions at error quad points')

t_start = default_timer()
if reuseGP:
    SHPE = SHP
    SHPdX0E = SHPdX0
    SHPdX1E = SHPdX1
    
else:
    eval_ptsE = np.vstack((gpX0E, gpX1E)).T
    SHPE, SHPdX0E, SHPdX1E = RKPMBasis.evalSHP(eval_ptsE,returnFirstDerivative=True)

t_stop = default_timer()
t_ESHP_comp = t_stop - t_start
############# End compute SHP functions at error calc quadrature points ##################
print('Computing errors')

t_start = default_timer()

gpSolndX0 = (SHPdX0E@nodal_soln.T)
gpSolndX1 = (SHPdX1E@nodal_soln.T)
gpSolnGrad = np.hstack((gpSolndX0,gpSolndX0))
gpSoln = (SHPE@nodal_soln.T)
coords = np.vstack((gpX0E,gpX1E))
T_ex_v = T_ex(coords)
gradT_ex_v = gradT_ex(coords)

L2net = np.sum(gpWE*(np.square(T_ex_v) ))
L2error = np.sum(gpWE*(np.square(T_ex_v - gpSoln[:,0]) ))

H10net = gpWE*(gradT_ex_v[0].T)@(gradT_ex_v[0]) + gpWE*(gradT_ex_v[1].T)@(gradT_ex_v[1])
H10error = gpWE*((gradT_ex_v[0] - gpSolndX0.T))@((gradT_ex_v[0]- gpSolndX0.T).T) + gpWE*((gradT_ex_v[1] - gpSolndX1.T))@((gradT_ex_v[1]- gpSolndX1.T).T) 
L2errorNorm = (L2error/L2net)**0.5
H10errorNorm = (H10error/H10net)**0.5

t_stop = default_timer()
t_E_comp = t_stop - t_start

if write_file: 
    print('writing to file')
    print(output_file)
    f = open(output_file,'a')
    f.write("\n")
    fs =  str(nP)+","+str(ref_val)+","+str(n)+","+str(L2errorNorm)+","+str(H10errorNorm[0][0])\
            +","+kernel+","+str(supp) +","+str(eps_val)+","+str(nGP)+","+str(gcPerEdgeC) +","+str(nGPE)+","+str(gcPerEdgeCE)\
            +","+str(t_SHP_comp) +","+str(t_assemble) +","+str(t_solve) +","+str(t_ESHP_comp) +","+str(t_E_comp)
    f.write(fs)
    f.close()


print(f"L2 error : {L2errorNorm}")
print(f"H10 error: {H10errorNorm[0][0]}")
print(f"Time to precompute SHP at GQ points: {t_SHP_comp}")
print(f"Time to assemble matrices: {t_assemble}")
print(f"Time to solve linear system : {t_solve}")
print(f"Time to precompute SHP at error GQ points: {t_ESHP_comp}")
print(f"Time to compute error: {t_E_comp}")


if visOutput:
    folder = 'poissonPics/'
    fig1, ax1 = plt.subplots()
    tcf = ax1.tricontourf(gpX0E, gpX1E, gpSoln[:,0])
    fig1.colorbar(tcf)
    fig1.savefig(folder + 'solution.png')

    fig2, ax2 = plt.subplots()
    tcf = ax2.tricontourf(gpX0E, gpX1E, T_ex_v)
    fig2.colorbar(tcf)
    fig2.savefig(folder + 'exact.png')

    fig3, ax3= plt.subplots()
    tcf = ax3.tricontourf(gpX0E, gpX1E, (np.multiply((T_ex_v - gpSoln[:,0]),(T_ex_v - gpSoln[:,0]))))
    fig3.colorbar(tcf)
    fig3.savefig(folder + 'error.png')

    fig4, ax4 = plt.subplots()
    tcf = ax4.tricontourf(gpX0E, gpX1E, gpSolndX0[:,0])
    fig4.colorbar(tcf)
    fig4.savefig(folder + 'dX0solution.png')

    fig5, ax5 = plt.subplots()
    tcf = ax5.tricontourf(gpX0E, gpX1E, gradT_ex_v[0])
    fig5.colorbar(tcf)
    fig5.savefig(folder + 'dX0exact.png')

    fig6, ax6= plt.subplots()
    tcf = ax6.tricontourf(gpX0E, gpX1E, (np.multiply((gradT_ex_v[0] - gpSolndX0[:,0]),(gradT_ex_v[0] - gpSolndX0[:,0]))))
    fig6.colorbar(tcf)
    fig6.savefig(folder + 'dX0error.png')