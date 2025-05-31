'''
A script to solve a multi-material poisson problem on a 1x1 unit square with RKPM, using guass quadrature 

Author: Jennifer E Fromm

'''

import numpy as np 
from IntRKPM import linAlgHelp, classicRKPMUtils
import os.path
import matplotlib.pyplot as plt 
import os
from timeit import default_timer
from scipy import sparse, spatial
import itertools
 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--wf',dest='wf',default=False,
                    help='Write error to file (True/False)')
parser.add_argument('--vd',dest='vd',default=False,
                    help='visualize data by generating mesh files with solution, default False')
parser.add_argument('--of',dest='of',default="poissonMM.csv",
                    help='Output file destination')
parser.add_argument('--n',dest='n',default=1,
                    help='RK polynomial order')
parser.add_argument('--ref',dest='ref',default=0,
                    help='Refinement level')
parser.add_argument('--supp',dest='supp',default=None,
                    help='Support size, default n + 1')
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

eps_val = float(args.eps)
opt = args.opt

ref_dir = 'R' + str(ref_val) + '/'
corner0 = [0.0,0.0]
corner1 = [1.0,1.0]

# interfaces at L/5 and 4*L/5 
L = 1
aveKernel = L/ (5*2**ref_val)
def indicator0(coords):
    return (np.atleast_2d(np.less_equal(coords[:,0], L/5))).astype(int)
def indicator1(coords):
    return (np.atleast_2d(np.logical_and(np.greater(coords[:,0], L/5), np.less(coords[:,0], 4*L/5)))).astype(int)
def indicator2(coords):
    return (np.atleast_2d(np.greater_equal(coords[:,0], 4*L/5))).astype(int)

#pointwise indicator functions 
def ptws_indicator0(coord):
    return coord[0] <= L/5
def ptws_indicator1(coord):
    return ((coord[0] >  L/5) and (coord[0]<= 4*L/5))
def ptws_indicator2(coord):
    return coord[0] > 4*L/5


eps = L*1e-12
def dist0(coords):
    return np.atleast_2d(np.add(np.subtract((L/5),coords[:,0]),eps))
def dist1(coords):
    return np.atleast_2d(np.minimum(np.subtract(np.subtract(coords[:,0],(L/5)),eps), np.subtract(np.subtract((4*L/5),coords[:,0]),eps)))
def dist2(coords):
    return np.atleast_2d(np.subtract(np.subtract(coords[:,0],(4*L/5)),eps))

def onInterface(coords):
    return np.nonzero(np.logical_or(np.isclose(coords[:,0],L/5),np.isclose(coords[:,0],4*L/5)))

kernel = 'SPLIN3'
def kernelFunc(z): 
    case1 = np.less_equal(z, 0.5)
    w1 = 2/3 - 4*z**2 + 4*z**3
    dwdr1 = -8*z + 12*z**2
    case2 = np.logical_and(np.less_equal(z,1.0), np.greater(z,0.5))
    w2= 4/3 - 4*z + 4*z**2 - (4*z**3)/3
    dwdr2 = -4 + 8*z -4*z**2
    w = np.multiply(case1,w1) + np.multiply(case2,w2)
    dwdr = np.multiply(case1,dwdr1) + np.multiply(case2,dwdr2)
    return w, dwdr



if opt == '_':
    print( 'No enrichment')
    enrich = False
elif opt == 'enrich':
    print( 'Using Heaviside enrichment')
    enrich = True
else:
    print("scaling option not supported")
    exit() 




RKPMBasis = classicRKPMUtils.RKPMBasisGenerator(ref_val)
def norm(coord0,coord1):
    return np.sqrt((coord0[0] - coord1[0])**2 + (coord0[1] - coord1[1])**2)

t_file = "RKPMPoints/" + ref_dir + "tri.xdmf"

corner0 = [0.0,0.0]
corner1 = [1.0,1.0]
corners = [corner0,corner1]
edgenodes = 5
mesh_exists=os.path.isfile(t_file) 
if mesh_exists:
    RKPMBasis.readPoints(t_file)
else:
    print("warning: generating new point set")
    RKPMBasis.makePoints(gridType=gridType, meshEps=eps_val,corners = corners, edgeNodes=edgenodes)
    RKPMBasis.savePoints(t_file)
RKPMBasis.makeBasis(kernelType=kernel,polynomialOrder=n,supportSize=supp,numberNeighbors=nnn,supportType=supportType)


matIndicators = [indicator0(RKPMBasis.nodeCoords), indicator1(RKPMBasis.nodeCoords), indicator2(RKPMBasis.nodeCoords)]

if enrich:
    # identify nodes that overlap with boundary:
    dist01 = np.abs(RKPMBasis.nodeCoords[:,0] - (L/5))
    dist12 = np.abs(RKPMBasis.nodeCoords[:,0] - (4*L/5))
    interface01_nodes =np.nonzero(np.less_equal(dist01, RKPMBasis.domainKernel))[0]
    multi_int_nodes = np.nonzero(np.logical_and(np.less_equal(dist01, RKPMBasis.domainKernel), np.less_equal(dist12, RKPMBasis.domainKernel)))

    n_new_1_nodes = len(interface01_nodes)
    new_1_nodes = np.arange(RKPMBasis.nP, (n_new_1_nodes+RKPMBasis.nP))

    interface12_nodes = np.nonzero(np.less_equal(dist12, RKPMBasis.domainKernel))[0]
    n_new_2_nodes = len(interface12_nodes)
    new_2_nodes = np.arange((n_new_1_nodes+RKPMBasis.nP), (n_new_2_nodes+n_new_1_nodes+RKPMBasis.nP))

    new_nP = n_new_2_nodes+n_new_1_nodes+RKPMBasis.nP
    # columns correspond to phase- make a new list of nodes where phase ID corresponds to enrichment func 
    # new matIndicator Functions to be used for enrichment 
    original_nodes = np.arange(0,RKPMBasis.nP)
    mat0enrich = np.zeros(new_nP)
    mat1enrich = np.zeros(new_nP)
    mat2enrich = np.zeros(new_nP)
    # first set to original mat indicators - will take care of nodes that are not interface nodes
    mat0enrich[original_nodes] = matIndicators[0]
    mat1enrich[original_nodes] = matIndicators[1]
    mat2enrich[original_nodes] = matIndicators[2]
    # add in nodes that are in mat 1 or mat 2  but enriched so their IDS now belong to mat 0 
    mat0enrich[interface01_nodes] = np.ones_like(interface01_nodes)
    # add in nodes that are in mat 2 but enriched so their IDS now belong to mat 1
    mat1enrich[interface12_nodes] = np.ones_like(interface12_nodes)

    # remove 01 interface nodes from mat 1 and mat 2
    mat1enrich[interface01_nodes] = np.zeros_like(interface01_nodes)
    mat2enrich[interface01_nodes] = np.zeros_like(interface01_nodes)
    
    # remove 12 interface nodes from mat 2 and mat 0? 
    mat2enrich[interface12_nodes] = np.zeros_like(interface12_nodes)

    # add in new nodes to appropriate material 
    mat1enrich[new_1_nodes] = np.ones_like(new_1_nodes)
    mat2enrich[new_2_nodes] = np.ones_like(new_2_nodes)

    # need new coord list    # need new domain kernel list 
    nodeCoords = np.vstack((RKPMBasis.nodeCoords, RKPMBasis.nodeCoords[interface01_nodes],  RKPMBasis.nodeCoords[interface12_nodes]))
    domainKernel = np.hstack((RKPMBasis.domainKernel, RKPMBasis.domainKernel[interface01_nodes],  RKPMBasis.domainKernel[interface12_nodes]))
else:
    nodeCoords = RKPMBasis.nodeCoords
    domainKernel = RKPMBasis.domainKernel
    new_nP = RKPMBasis.nP
    

 

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



#create interface nodes, stack them in order
#gamma0 - left
gamma01 = np.vstack((np.ones_like(gcVerts1)*(L/5),gcVerts1)).T
gamma12 = np.vstack((np.ones_like(gcVerts1)*(4*L/5),gcVerts1)).T
gammaInts = [gamma01, gamma12]
IntsgpX0 = []
IntsgpX1 = []
IntsgpW = []
IntsgpNormals = []
nsGPInt = []
for gammaID in range(2):
    gammaInt = gammaInts[gammaID]
    IntgpX0 = []
    IntgpX1 = []
    IntgpW = []
    IntgpNormals = None
    for i in range(gammaInt.shape[0] -1):
        coord0 = gammaInt[i,:]
        coord1 = gammaInt[i+1,:]
        X0,_ = classicRKPMUtils.getGaussQuad(nGP, coord0[0], coord1[0])
        X1,_ = classicRKPMUtils.getGaussQuad(nGP, coord0[1], coord1[1])
        L_seg = norm(coord0,coord1)
        _,W = classicRKPMUtils.getGaussQuad(nGP, 0.0, L_seg)
        X10 = coord1 - coord0 
        X10_normal = np.matmul(X10,rotateMat)
        X10_normal = X10_normal / (norm(X10_normal, [0,0]))
        gpNormals = np.tile(X10_normal,(nGP,1))

        IntgpX0 = np.hstack((IntgpX0,X0))
        IntgpX1 = np.hstack((IntgpX1,X1))
        IntgpW = np.hstack((IntgpW,W))
        if IntgpNormals is not None:
            IntgpNormals= np.vstack((IntgpNormals,gpNormals))
        else:
            IntgpNormals = gpNormals 
    IntsgpX0 += [IntgpX0]
    IntsgpX1 += [IntgpX1]
    IntsgpW += [IntgpW]
    IntsgpNormals += [IntgpNormals]

    nsGPInt += [IntgpW.size]

############# End compute quadrature points   ##################


############# Compute SHP at quadrature points   ##################
print('Computing shape functions at quad points')
t_start = default_timer()


eval_pts = np.vstack((gpX0, gpX1)).T
onInterfaceVal = onInterface(RKPMBasis.nodeCoords)[0]

SHP, SHPdX0, SHPdX1 = RKPMBasis.evalSHP(eval_pts)

BCeval_pts = np.vstack((BCgpX0, BCgpX1)).T
BC_SHP, BC_SHPdX0, BC_SHPdX1 = RKPMBasis.evalSHP(BCeval_pts)


if enrich:
    print('computing interface SHP funcs')
    Inteval_pts01 = np.vstack((IntsgpX0[0], IntsgpX1[0])).T
    Int01_SHP, Int01_SHPdX0, Int01_SHPdX1 = RKPMBasis.evalSHP(Inteval_pts01)

    Inteval_pts12 = np.vstack((IntsgpX0[1], IntsgpX1[1])).T
    Int12_SHP, Int12_SHPdX0, Int12_SHPdX1 = RKPMBasis.evalSHP(Inteval_pts12)


t_stop = default_timer()
t_SHP_comp = t_stop - t_start

t_start = default_timer()
if enrich:
    print('Enriching shape functions at quad points')
    # loop over each set of eval pts (for bulk, boundary, and each interface)
    pointSets = [eval_pts,BCeval_pts,Inteval_pts01,Inteval_pts12]#,Inteval_pts01_left,Inteval_pts01_right,Inteval_pts12_left,Inteval_pts12_right]
    SHPs = [SHP, BC_SHP,Int01_SHP,Int12_SHP]#Int01_SHP_0,Int01_SHP_1,Int12_SHP_1,Int12_SHP_2]
    SHPdX0s= [ SHPdX0, BC_SHPdX0,Int01_SHPdX0, Int12_SHPdX0]#, Int01_SHPdX0_0,Int01_SHPdX0_1,Int12_SHPdX0_1,Int12_SHPdX0_2]
    SHPdX1s= [ SHPdX1, BC_SHPdX1,Int01_SHPdX1, Int12_SHPdX1]
    notInt = [True, True, False, False]
    enrichedSHPs = []
    enrichedSHPdX0s = []
    enrichedSHPdX1s = []
    eval_ID = 0 
    for eval_pts in pointSets:
        n_pts= eval_pts.shape[0]
        ogSHP = SHPs[eval_ID]
        ogSHPdX0 = SHPdX0s[eval_ID]
        ogSHPdX1 = SHPdX1s[eval_ID]
        eSHP = sparse.lil_array((n_pts,new_nP))
        eSHPdX0 = sparse.lil_array((n_pts,new_nP))
        eSHPdX1 = sparse.lil_array((n_pts,new_nP))

        # add new node evaluations
        eSHP[:,original_nodes] = ogSHP
        eSHP[:,new_1_nodes] = ogSHP[:,interface01_nodes]
        eSHP[:,new_2_nodes] = ogSHP[:,interface12_nodes]

        eSHPdX0[:,original_nodes] = ogSHPdX0
        eSHPdX0[:,new_1_nodes] = ogSHPdX0[:,interface01_nodes]
        eSHPdX0[:,new_2_nodes] = ogSHPdX0[:,interface12_nodes]

        eSHPdX1[:,original_nodes] = ogSHPdX1
        eSHPdX1[:,new_1_nodes] = ogSHPdX1[:,interface01_nodes]
        eSHPdX1[:,new_2_nodes] = ogSHPdX1[:,interface12_nodes]

        # zero out regions 
        if notInt[eval_ID]:
            pt_ID = 0 
            for pt in eval_pts:
                pt_enrich = mat0enrich* ptws_indicator0(pt) + mat1enrich* ptws_indicator1(pt) + mat2enrich* ptws_indicator2(pt)
                eSHP[[pt_ID], :] = pt_enrich* eSHP[[pt_ID], :]
                eSHPdX0[[pt_ID], :] = pt_enrich* eSHPdX0[[pt_ID], :]
                eSHPdX1[[pt_ID], :] = pt_enrich* eSHPdX1[[pt_ID], :]
                pt_ID += 1

        # store in enriched SHPs 
        enrichedSHPs += [eSHP.tocsr()]
        enrichedSHPdX0s += [eSHPdX0.tocsr()]
        enrichedSHPdX1s += [eSHPdX1.tocsr()]
        
        eval_ID += 1

    SHP, BC_SHP, Int01_SHP,Int12_SHP= enrichedSHPs
    SHPdX0, BC_SHPdX0, Int01_SHPdX0,Int12_SHPdX0 = enrichedSHPdX0s
    SHPdX1, BC_SHPdX1, Int01_SHPdX1,Int12_SHPdX1 = enrichedSHPdX1s


t_stop = default_timer()
t_enrich_comp = t_stop - t_start

############# End compute SHP at quadrature points   ##################

############# Assemble matrices  ##################
print('Assembling matrices')    
# solving simple poisson's problem Kx=b
    
kappa1 = 1.0
kappa2 = 0.5
kappa3 = 1.0
def kappaFunc(x):
    if x[0] <= L/5:
        kappa = kappa1
    elif x[0] <= 4*L/5:
        kappa = kappa2
    else:
        kappa = kappa3
    return kappa
def kappaVec(x):
    mat1 = np.less_equal(x[0], L/5)
    mat2 =  np.logical_and(np.greater(x[0],L/5), np.less_equal(x[0],4*L/5))
    mat3 = np.greater(x[0],4*L/5)
    return kappa1*mat1 + kappa2*mat2 + kappa3*mat3


beta_param = 10
beta = beta_param/ domainKernel 
sgn = -1.0 # for nonsymmetric 

def T_ex(x): 
    x_bar = x[0] - (L/5)
    return np.sin(5*np.pi*x[1]/(3*L))* np.sin(5*np.pi*x_bar/(3*L))/kappaFunc(x)

def T_exNoKappa(x): 
    x_bar = x[0] - (L/5)
    return np.sin(5*np.pi*x[1]/(3*L))* np.sin(5*np.pi*x_bar/(3*L))

def gradT_ex(x): 
    x_bar = x[0] - (L/5)
    dx0 = np.sin(5*np.pi*x[1]/(3*L))*(5*np.pi/(3*L))*np.cos(5*np.pi*x_bar/(3*L))/kappaFunc(x)
    dx1 = np.cos(5*np.pi*x[1]/(3*L))*(5*np.pi/(3*L))*np.sin(5*np.pi*x_bar/(3*L))/kappaFunc(x)
    return np.vstack((dx0,dx1))
def gradT_exNoKappa(x): 
    x_bar = x[0] - (L/5)
    dx0 = np.sin(5*np.pi*x[1]/(3*L))*(5*np.pi/(3*L))*np.cos(5*np.pi*x_bar/(3*L))
    dx1 = np.cos(5*np.pi*x[1]/(3*L))*(5*np.pi/(3*L))*np.sin(5*np.pi*x_bar/(3*L))
    return np.vstack((dx0,dx1))


def f_b(x):
    # f_b = - kappa* div(grad(T_ex))
    x_bar = x[0] - (L/5)
    return (25*np.pi*np.pi/(9*L*L))*( 2*np.sin(5*np.pi*x_bar/(3*L))*np.sin(5*np.pi*x[1]/(3*L)))

#bulk
# KIJ = : grad(T)* grad(theta)* kappa
#   B'B kappa
# F_b = theta* f_b(qpt)
#   Psi' f_b(qpt)
t_start = default_timer()

nP = new_nP
coords = np.vstack((gpX0,gpX1))
f_b_v = f_b(coords)
kappa_v = kappaVec(coords)
B_all =sparse.lil_array((2*nGPBulk,nP))
B_all[::2] = SHPdX0
B_all[1::2] = SHPdX1 
W_2 =np.zeros((2*nGPBulk,1))
W_2[::2] = np.asarray([gpW]).T
W_2[1::2] = np.asarray([gpW]).T

kappa_v_B =np.zeros((2*nGPBulk,1))
kappa_v_B[::2] = np.asarray([kappa_v]).T
kappa_v_B[1::2] = np.asarray([kappa_v]).T

KIJ = (kappa_v_B*B_all).T@ (W_2*B_all)
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
#kappa_v_BC = np.piecewise(BCgpX0, [BCgpX0<= (L/5), np.logical_and(BCgpX0> (L/5), BCgpX0<= (4*L/5)), BCgpX0> (4*L/5)], [kappa1, kappa2, kappa3])
kappa_v_BC = kappaVec(BCcoords)

BCT_ex_noKappa = T_exNoKappa(BCcoords)
BCT_ex_v = np.divide(BCT_ex_noKappa, kappa_v_BC)

#note- could be combined to save memory, left as separate to help with debugging 
F_pen_v = (BCgpW.T*BCT_ex_v)@BC_SHP* beta
F_const_v = sgn*(np.multiply(kappa_v_BC,BCgpW).T*BCT_ex_v)@(norm_mat@BC_B_all)
F += F_pen_v + F_const_v

K_pen_v = (beta*BC_SHP).T@((np.asarray([BCgpW]).T*BC_SHP))
K_const_v = (np.asarray([kappa_v_BC]).T*((np.asarray([BCgpW]).T*BC_SHP))).T@(norm_mat@BC_B_all)
K_Aconst_v = sgn* K_const_v.T

KIJ += K_pen_v + K_const_v + K_Aconst_v


# nitsches interface conditions
# R =  -({kappa \grad(theta)}dot n)[[T]]dGamma \
#      - ([[theta]]{kappa \grad(T)}dot n)dGamma \
#      + (gamma [[theta]][[T]]) dGamma
# first - pure penalty at interface 

if enrich:
    print('adding Nitsches interface terms')
    # use diff coords 
    jump01 = mat0enrich- mat1enrich
    jump12 = mat1enrich- mat2enrich
    #jump01 = mat1enrich- mat0enrich
    #jump12 = mat2enrich- mat1enrich
    add01 = mat0enrich +  mat1enrich
    add12 = mat1enrich + mat2enrich
    kappa_nodes_v = kappa1*mat0enrich + kappa2*mat1enrich + kappa3*mat2enrich
  
    domain_nodes_v = np.atleast_2d(domainKernel)
    gamma_denom_v = np.divide(np.square(domain_nodes_v), kappa_nodes_v) 
    gamma_denom_mat = gamma_denom_v.T +  gamma_denom_v 
    gamma_num_mat = domain_nodes_v.T + domain_nodes_v
    gamma_const = 100
    gamma_mat = 2*gamma_const* np.divide(gamma_num_mat, gamma_denom_mat)

    gamma_mat = (gamma_const)*(5* 2**ref_val)

    K_pen01 = gamma_mat*(((( Int01_SHP*jump01)).T )@ (np.asarray([IntsgpW[0]]).T*(Int01_SHP*jump01)))
    K_pen12 = gamma_mat*(((( Int12_SHP*jump12)).T )@ (np.asarray([IntsgpW[1]]).T*(Int12_SHP*jump12)))
    #

    ave_denom_mat = gamma_denom_mat
    ave_num_vec= np.atleast_2d(np.divide(domain_nodes_v, kappa_nodes_v))
    ave_num_vec= np.atleast_2d(domain_nodes_v)

    ave01 = add01/((1/kappa1) + (1/kappa2))
    ave12 = add12/((1/kappa2) + (1/kappa3))

    Int01_B_all =sparse.lil_array((2*nsGPInt[0],nP))
    Int01_B_all[::2] = Int01_SHPdX0
    Int01_B_all[1::2] = Int01_SHPdX1 

    Int01_norm_mat1 = np.diagflat(IntsgpNormals[0][:,0])
    Int01_norm_mat2 = np.diagflat(IntsgpNormals[0][:,1])
    Int01_norm_mat = np.zeros((nsGPInt[0],2*nsGPInt[0]))
    Int01_norm_mat[:,::2] = Int01_norm_mat1
    Int01_norm_mat[:,1::2] = Int01_norm_mat2

    K_const01 = (( Int01_SHP*jump01).T * np.asarray([IntsgpW[0]])) @ (Int01_SHPdX0*ave01)
    K_Aconst01 = K_const01.T
    


    Int12_B_all =sparse.lil_array((2*nsGPInt[1],nP))
    Int12_B_all[::2] = Int12_SHPdX0
    Int12_B_all[1::2] = Int12_SHPdX1 

    Int12_norm_mat1 = np.diagflat(IntsgpNormals[1][:,0])
    Int12_norm_mat2 = np.diagflat(IntsgpNormals[1][:,1])
    Int12_norm_mat = np.zeros((nsGPInt[1],2*nsGPInt[1]))
    Int12_norm_mat[:,::2] = Int12_norm_mat1
    Int12_norm_mat[:,1::2] = Int12_norm_mat2

    K_const12 = (( Int12_SHP*jump12).T * np.asarray([IntsgpW[1]])) @ (Int12_SHPdX0*ave12)
    K_Aconst12 = K_const12.T

    sgn_pen = 1
    sgn_const = -1
    sgn_aconst = 1
    KIJ += sgn_pen*(K_pen01 + K_pen12)
    KIJ += sgn_const*(K_const01) + sgn_aconst*(K_Aconst01)
    KIJ += sgn_const*(K_const12) + sgn_aconst*(K_Aconst12)


'''
# L2 projection - can help with debugging
KIJ = ((np.atleast_2d(gpW).T)*SHP).T@(SHP)
coords = np.vstack((gpX0,gpX1))
kappa_v = kappaVec(coords)
T_ex_noKappa = T_exNoKappa(coords)
T_ex_v = np.divide(T_ex_noKappa, kappa_v)
F = (gpW.T*T_ex_v)@SHP
'''

t_stop = default_timer()
t_assemble = t_stop - t_start


############# End assemble matrices  ##################
    
############# Solve linear system  ##################    
print('Solving linear system ')
# convert to petsc so I can use my solvers :) 
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
    SHPE, SHPdX0E, SHPdX1E = RKPMBasis.evalSHP(eval_ptsE)

t_stop = default_timer()
t_ESHP_comp = t_stop - t_start
############# End compute SHP functions at error calc quadrature points ##################
print('Computing errors')

t_start = default_timer()

gpSolndX0 = (SHPdX0E@nodal_soln.T)
gpSolndX1 = (SHPdX1E@nodal_soln.T)
gpSolnGrad = np.hstack((gpSolndX0,gpSolndX1))
gpSoln = (SHPE@nodal_soln.T)
coords = np.vstack((gpX0E,gpX1E))
kappa_vE = kappaVec(coords)
T_ex_noKappa = T_exNoKappa(coords)
T_ex_v = np.divide(T_ex_noKappa, kappa_vE)
gradT_ex_noKappa = gradT_exNoKappa(coords)
gradT_ex_v = np.divide(gradT_ex_noKappa, kappa_vE)

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
    fs =  str(new_nP)+","+opt+","+str(ref_val)+","+str(n)+","+str(L2errorNorm)+","+str(H10errorNorm[0][0])\
            +","+kernel+","+str(supp) +","+str(eps_val)+","+str(nGP)+","+str(gcPerEdgeC) +","+str(nGPE)+","+str(gcPerEdgeCE)\
            +","+str(t_SHP_comp) +","+str(t_assemble) +","+str(t_solve) +","+str(t_ESHP_comp) +","+str(t_E_comp) +","+str(t_enrich_comp)
    f.write(fs)
    f.close()


print(f"L2 error : {L2errorNorm}")
print(f"H10 error: {H10errorNorm[0][0]}")
print(f"Time to precompute SHP at GQ points: {t_SHP_comp}")
print(f"Time to enrich SHP at all GQ points: {t_enrich_comp}")
print(f"Time to assemble matrices: {t_assemble}")
print(f"Time to solve linear system : {t_solve}")
print(f"Time to precompute SHP at error GQ points: {t_ESHP_comp}")
print(f"Time to compute error: {t_E_comp}")


if visOutput:
    # plot solution 

    pltRes = 100*(2**ref_val)
    plotPtsX1 = 0.3*np.ones(pltRes)
    plotPtsX0 = np.linspace(0.0, 1.0, pltRes)

    plt_pts = np.vstack((plotPtsX0, plotPtsX1)).T
    pltSHP, pltSHPdX0, pltSHPdX1 = RKPMBasis.evalSHPSemiVecSparse(plt_pts,kernelFunc,matIndicators=matIndicators, onInterfaceVal=onInterfaceVal, scalingFunc = scalingFuncP, indicatorFunc = indicatorFuncP, aveKernel=aveKernel, implicit=False)

    if enrich:
        n_pts = plt_pts.shape[0]
        eSHP = sparse.lil_array((n_pts,new_nP))
        eSHPdX0 = sparse.lil_array((n_pts,new_nP))
        eSHPdX1 = sparse.lil_array((n_pts,new_nP))

        # add new node evaluations
        eSHP[:,original_nodes] = pltSHP
        eSHP[:,new_1_nodes] = pltSHP[:,interface01_nodes]
        eSHP[:,new_2_nodes] = pltSHP[:,interface12_nodes]

        eSHPdX0[:,original_nodes] = pltSHPdX0
        eSHPdX0[:,new_1_nodes] = pltSHPdX0[:,interface01_nodes]
        eSHPdX0[:,new_2_nodes] = pltSHPdX0[:,interface12_nodes]

        eSHPdX1[:,original_nodes] = pltSHPdX1
        eSHPdX1[:,new_1_nodes] = pltSHPdX1[:,interface01_nodes]
        eSHPdX1[:,new_2_nodes] = pltSHPdX1[:,interface12_nodes]

        # zero out regions 
        pt_ID = 0 
        for pt in plt_pts:
            pt_enrich = mat0enrich*ptws_indicator0(pt) + mat1enrich*ptws_indicator1(pt) + mat2enrich*ptws_indicator2(pt)
            eSHP[[pt_ID], :] = pt_enrich* eSHP[[pt_ID], :]
            eSHPdX0[[pt_ID], :] = pt_enrich* eSHPdX0[[pt_ID], :]
            eSHPdX1[[pt_ID], :] = pt_enrich* eSHPdX1[[pt_ID], :]
            pt_ID += 1
        
        pltSHP = eSHP.tocsr()
        pltSHPdX0 = eSHPdX0.tocsr()
        pltSHPdX1 = eSHPdX1.tocsr()




    pltSolndX0 = (pltSHPdX0@nodal_soln.T)
    pltSolndX1 = (pltSHPdX1@nodal_soln.T)
    pltSoln = (pltSHP@nodal_soln.T)

    kappa_vplt = kappaVec(plt_pts.T)
    T_ex_noKappa = T_exNoKappa(plt_pts.T)
    T_ex_v = np.divide(T_ex_noKappa, kappa_vplt)
    gradT_ex_noKappa = gradT_exNoKappa(plt_pts.T)
    gradT_ex_v = np.divide(gradT_ex_noKappa, kappa_vplt)


    from matplotlib import pyplot as plt
    fig, axes= plt.subplots(4, 1, figsize=(12, 12))

    axes[0].plot(plotPtsX0,T_ex_v, label = 'exact', linestyle= 'dotted')
    axes[0].plot(plotPtsX0,pltSoln, label = 'solution')

    axes[1].plot(plotPtsX0,np.abs((T_ex_v -pltSoln[:,0])) , label = 'error')

    axes[2].plot(plotPtsX0,gradT_ex_v[0], label = 'exact', linestyle= 'dotted')
    axes[2].plot(plotPtsX0,pltSolndX0, label = 'solution')


    axes[3].plot(plotPtsX0,np.abs((gradT_ex_v[0] -pltSolndX0[:,0])), label = 'error')

    y_labels = ['solution', 'solution error', 'dx', 'dx error'] 
    axe_count = 0
    for ax in axes:
        ax.legend( fontsize=12,frameon=False)#, loc='lower right')
        ax.set_xlabel('x', fontsize='x-large')
        ax.set_ylabel(y_labels[axe_count], fontsize='x-large')
        #ax.set_title(axis_titles[axe_count],fontsize='xx-large')
        axe_count += 1


    fig.suptitle('Solution along Y=0.3',fontsize='xx-large')
    plt.tight_layout()

    fig_file = 'file.png'


    plt.savefig(fig_file)

    import pandas as pd
    # save data as csv w/ dataframe
    data = {'x0': plotPtsX0, 'x1': plotPtsX1, 'T_ex': T_ex_v, 'dx0_ex': gradT_ex_v[0],'dx1_ex': gradT_ex_v[1],
            'T': pltSoln[:,0], 'dx0': pltSolndX0[:,0], 'dx1': pltSolndX1[:,0]}

    df = pd.DataFrame(data)
    df_file = 'data.csv' 
    df.to_csv(df_file, index=False)



    folder = 'solutionPics/'
    fig1, ax1 = plt.subplots()
    tcf = ax1.tricontourf(gpX0E, gpX1E, gpSoln[:,0])
    fig1.colorbar(tcf)
    fig1.savefig(folder + 'solution.png')

    fig2, ax2 = plt.subplots()
    tcf = ax2.tricontourf(gpX0E, gpX1E, T_ex_v)
    fig2.colorbar(tcf)
    fig2.savefig(folder + 'exact.png')

    fig3, ax3= plt.subplots()
    #tcf = ax3.tricontourf(gpX0, gpX1, gpEx - gpSoln)
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
    #tcf = ax3.tricontourf(gpX0, gpX1, gpEx - gpSoln)
    tcf = ax6.tricontourf(gpX0E, gpX1E, (np.multiply((gradT_ex_v[0] - gpSolndX0[:,0]),(gradT_ex_v[0] - gpSolndX0[:,0]))))
    fig6.colorbar(tcf)
    fig6.savefig(folder + 'dX0error.png')