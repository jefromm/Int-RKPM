import numpy as np
from scipy import spatial, sparse
#note- meshio only required to save and reuse point clouds
import meshio
import math
import abc
from IntRKPM import kernels

def getGaussQuad(n,a,b):
    '''
    compute gaussian integrations points for a 1-D domain
    inputs:
    n: number of points 
    a: left-hand coordinate
    b: right-hand coordinate

    outputs:

    xx: point coordinates
    ww: point weights

    '''
    u = np.arange(1,n)
    u = u/ (np.sqrt(4*u**2 - 1))

    A1 = np.diagflat(u,-1)
    A2 = np.diagflat(u,1)
    A = A1 + A2

    x,v = np.linalg.eig(A)
    k = np.argsort(x)
    x = np.sort(x)

    w = 2 *v[0,k]**2 

    # linear transformation 
    x = 0.5* (b-a) *x + 0.5*(a + b)
    w = 0.5* (b-a) *w
    return x, w 
    


class RKPMBasisGenerator(object):

    '''
    Abstract class to construct an RKPM basis
    that can be evaluated at arbitrary points
    '''
    __metaclass__ = abc.ABCMeta

    def __init__(self,*args):

        """
        Arguments in ``*args`` are passed as a tuple to
        ``self.customSetup()``.  Appropriate arguments vary by subclass.
        """
        self.setup(args)

    def setup(self,args):
            self.refVal = args[0]

    def makePoints(self,gridType='jitter',meshEps=0.5,corners=[[0,0],[1,1]],edgeNodes=[10,10],fg_corners=None,addBoundaryNodes=False):
        """
        Function to create point cloud for RKPM nodes

        Parameters
        ----------
        gridType:str,  "jitter" or "random"
        meshEps:flt in (0,1)
        corners:corners of rectangular domain
        edgenodes:int or [int,int]
        fg_corners:boolean, option to state foreground domain corners, used for adding buffer region in immersed setting 
        addBoundaryNodes:boolean 
        """
        
        if isinstance(edgeNodes, int):
            # if only one number is given, use that number for each edge
            edgeNodes= [edgeNodes, edgeNodes]
        self.edgeNodes = edgeNodes
        self.gridCorners = corners
        self.fgCorners = fg_corners
        if gridType == 'random':
            print("using random point distribution")
            self.genRandomNodes()
        elif gridType == 'jitter':
            self.meshEps=meshEps
            if self.fgCorners is not None:
                self.genNodesBuffer()
            else:
                self.genNodes()
        else:
            print('grid type not supported')
            exit()
        if addBoundaryNodes:
            if self.fgCorners is None:
                print('Caution: no foreground corners given, using grid corners instead')
                self.fgCorners = self.gridCorners
            self.genBoundaryNodes()

        self.x0I = self.nodeCoords[:,0]
        self.x1I = self.nodeCoords[:,1]


    def genRandomNodes(self):
        """
        Function to generate random point cloud
        """
        self.nP = ((2**self.refVal)*self.edgeNodes[0])*((2**self.refVal)*self.edgeNodes[1])
        cords_unscaled = np.random.rand(self.nP,2)
        xrange = self.gridCorners[1][0] - self.gridCorners[0][0]
        yrange = self.gridCorners[1][1] - self.gridCorners[0][1]
        cords_x =cords_unscaled* xrange + self.gridCorners[0][0]
        cords_y =cords_unscaled* yrange + self.gridCorners[0][1]
        cords  = np.vstack((cords_x,cords_y))
        self.nodeCoords = cords.T

    def genNodesBuffer(self):
        """
        Function to add region of buffer nodes outside the forground domain
        """
        numEdgeNodes = [ (self.edgeNodes[0]*(2**self.refVal) + 1), (self.edgeNodes[1]*(2**self.refVal) + 1)]
        self.nP = numEdgeNodes[0]*numEdgeNodes[1]
        self.x_spacing = (self.gridCorners[1][0]- self.gridCorners[0][0])/ numEdgeNodes[0]
        self.y_spacing = (self.gridCorners[1][1]- self.gridCorners[0][1])/ numEdgeNodes[1]
        x = np.linspace(self.gridCorners[0][0],self.gridCorners[1][0],numEdgeNodes[0])
        y = np.linspace(self.gridCorners[0][1],self.gridCorners[1][1],numEdgeNodes[1])
        xv,yv = np.meshgrid(x,y)
        perturbEps = self.meshEps
        # note: to prevent point over lap, must be less than node spacing
        sigmax = perturbEps*self.x_spacing
        sigmay = perturbEps*self.y_spacing
        x_perturb = np.array([(xv + sigmax* (np.random.rand(numEdgeNodes[1],numEdgeNodes[0]) - 0.5)).flatten()]).T
        y_perturb = np.array([(yv + sigmay* (np.random.rand(numEdgeNodes[1],numEdgeNodes[0 ]) - 0.5)).flatten()]).T
        self.nodeCoords = np.hstack((x_perturb,y_perturb))

        # get IDs of inner nodes
        if (self.gridCorners[0][0] > self.fgCorners[0][0]) or (self.gridCorners[0][1] > self.fgCorners[0][1]) \
            or (self.gridCorners[1][0] < self.fgCorners[1][0]) or (self.gridCorners[1][1] < self.fgCorners[1][1]):
            print('Error: Nodes do not span given foreground space')
            exit()
        
        insideXNodes_l = self.nodeCoords[(self.nodeCoords[:,0]>(self.fgCorners[0][0] - sigmax))]
        insideXNodes_r = insideXNodes_l[(insideXNodes_l[:,0]<(self.fgCorners[1][0] + sigmax))]
        insideYNodes_b = insideXNodes_r[(insideXNodes_r[:,1]>(self.fgCorners[0][1] - sigmay))]
        insideYNodes_t = insideYNodes_b[(insideYNodes_b[:,1]<(self.fgCorners[1][1] + sigmay))]
        self.innerNodeCoords = insideYNodes_t

    def genNodes(self):
        """
        Standard function for generating nodes in jittered grid
        """
        numEdgeNodes = [ (self.edgeNodes[0]*(2**self.refVal) + 1), (self.edgeNodes[1]*(2**self.refVal) + 1)]
        self.nP = numEdgeNodes[0]*numEdgeNodes[1]
        self.x_spacing = (self.gridCorners[1][0]- self.gridCorners[0][0])/ numEdgeNodes[0]
        self.y_spacing = (self.gridCorners[1][1]- self.gridCorners[0][1])/ numEdgeNodes[1]
        x = np.linspace(self.gridCorners[0][0],self.gridCorners[1][0],numEdgeNodes[0])
        y = np.linspace(self.gridCorners[0][1],self.gridCorners[1][1],numEdgeNodes[1])
        xv,yv = np.meshgrid(x,y)
        perturbEps = self.meshEps
        sigmax = perturbEps*self.x_spacing
        sigmay = perturbEps*self.y_spacing
        x_perturb = np.array([(xv + sigmax* (np.random.rand(numEdgeNodes[1],numEdgeNodes[0]) - 0.5)).flatten()]).T
        y_perturb = np.array([(yv + sigmay* (np.random.rand(numEdgeNodes[1],numEdgeNodes[0]) - 0.5)).flatten()]).T
        self.nodeCoords = np.hstack((x_perturb,y_perturb))


    def addNodes(self,coords):
        """
        Function to add nodes

        Parameters
        ----------
        gridType:str,  "jitter" or "random"

        """
        newnP = coords.shape[0]
        newx0I = coords[:,0]
        newx1I = coords[:,1]

        self.nP = self.nP + newnP
        self.x0I = np.hstack((self.x0I, newx0I))
        self.x1I = np.hstack((self.x1I, newx1I))
        self.nodeCoords = np.vstack((self.nodeCoords, coords))
        self.getTree()



    def genBoundaryNodes(self):
        """
        Function for adding nodes along the boundary of the initial domain
        """
        numXNodes = int(np.ceil((self.fgCorners[1][0]-self.fgCorners[0][0])/self.x_spacing))
        numYNodes = int(np.ceil((self.fgCorners[1][1]-self.fgCorners[0][1])/self.y_spacing))
        x_vec = np.linspace(self.fgCorners[0][0],self.fgCorners[1][0],numXNodes)[:,None]
        y_vec = np.linspace(self.fgCorners[0][1],self.fgCorners[1][1],numYNodes)[:,None]
        bottomNodes = np.hstack((x_vec,np.ones_like(x_vec)*self.fgCorners[0][1]))
        topNodes = np.hstack((x_vec,np.ones_like(x_vec)*self.fgCorners[1][1]))
        leftNodes = np.hstack((np.ones_like(y_vec)*self.fgCorners[0][0],y_vec))
        rightNodes = np.hstack((np.ones_like(y_vec)*self.fgCorners[1][0],y_vec))

        self.boundaryNodeCoords = np.vstack((bottomNodes,topNodes,leftNodes,rightNodes))


    def savePoints(self,t_file):
        """
        Function for saving points to file 

        Parameters
        ----------
        t_file:str,  file name 
        """
        tri = spatial.Delaunay(self.nodeCoords)
        meshio_mesh = meshio.Mesh(points=tri.points,cells=[('triangle', tri.simplices)])
        meshio.write(t_file,meshio_mesh)

    def readPoints(self,t_file,boundary_file = None):
        """
        Function for reading in nodes from file, with option to read in boundary nodes

        Parameters
        ----------
        t_file:             str,  file name 
        boundary_file:      str,  file
        """
        meshio_mesh = meshio.read(t_file)
        self.nodeCoords = meshio_mesh.points
        if boundary_file is not None:
            meshio_boundary_mesh = meshio.read(boundary_file)
            self.boundaryNodeCoords = meshio_boundary_mesh.points
            self.nodeCoords = np.vstack((self.boundaryNodeCoords,self.nodeCoords))
        self.nP = self.nodeCoords.shape[0]
        self.x0I = self.nodeCoords[:,0]
        self.x1I = self.nodeCoords[:,1]
    
    def makeBasis(self,kernelType='SPLIN3', polynomialOrder=1,supportSize=3,numberNeighbors=5,supportType='cir',augment=False):
        """
        Function to set up RKPM basis

        Parameters
        ----------
        polynomialOrder:        int
        supportSize:            flt
        numberNeighbors:        in
        supportType:            str ('cir' or 'rec')
        augment:                bool    

        """
        self.kernelType = kernelType
        self.kernelFunc = self.getKernelFunc(self.kernelType)
        self.n = polynomialOrder
        self.supportSize = supportSize
        self.k = numberNeighbors
        self.d = 2
        self.m =  int(math.factorial(self.n+self.d) / (math.factorial(self.n) * math.factorial(self.d)))
        if augment:
            # option to augment basis, ie to make bilinear/ biquadratic
            self.augmentBasis = True
            if self.n == 1:
                self.m = self.m + 1
            elif self.n == 2:
                self.m = self.m + 3
            elif self.n == 3:
                self.m = self.m + 6
        else:
            self.augmentBasis = False
        self.supportType = supportType
        self.getTree()


    def getTree(self):
        """
        Function to generate KD Tree to compute nodal spacing and kernel size
        """
        if self.supportType == 'cir':
            p = 2
        elif self.supportType == 'rec':
            p = np.inf
        else:
            print("support shape not suppported")
            exit()
        dd,_ = spatial.KDTree(self.nodeCoords).query(self.nodeCoords,k = (self.k+1), p=p)
        self.nodalSpacing = dd[:,self.k]
        self.domainKernel = self.nodalSpacing*self.supportSize


    def getKernelFunc(self,kType):
        ''' Function to specify the kernel function, based on the keyword'''
        if kType == 'SPLIN3':
            return kernels.SPLIN3
        else:
            print('Kernel function not yet supported')
            exit()



    def evalSHP(self, eval_pts,returnFirstDerivative=True, returnSecondDerivative=False):
        """
        Function to evaluate RKPM shape functions

        Parameters
        ----------
        eval_pts: coordinates of points
        returnFirstDerivative: bool
        returnSecondDerivative: bool
        """

        if returnSecondDerivative:
            returnFirstDerivative = True 

        n_pts = eval_pts.shape[0]
        x0 = eval_pts[:,0]
        x1 = eval_pts[:,1]

        #get node coords
        x0I = np.asarray([self.x0I]).T
        x1I = np.asarray([self.x1I]).T
        domainKernel = np.asarray([self.domainKernel]).T
        m = self.m
        # initialize arrays of shape functions and derivatives 
        psi = sparse.lil_array((n_pts,self.nP))
        dpsidx0 = sparse.lil_array((n_pts,self.nP))
        dpsidx1 = sparse.lil_array((n_pts,self.nP))
        dpsiddx0 = sparse.lil_array((n_pts,self.nP))
        dpsiddx1 = sparse.lil_array((n_pts,self.nP))
        dpsidx0dx1 = sparse.lil_array((n_pts,self.nP))


        # find  nodes in range
        # loop over points
        for i in range(n_pts):
            distMat = np.sqrt(np.square(np.subtract(x0[i], x0I)) + np.square(np.subtract(x1[i], x1I)))
            inRange = np.less_equal(distMat, domainKernel)
            x0_dist = np.multiply(np.subtract(x0[i], x0I),inRange)
            x1_dist = np.multiply(np.subtract(x1[i], x1I),inRange)

            # construct Moment matrices
            z = np.divide(distMat, domainKernel)
            phi, dwdz, dwddz= self.kernelFunc(z)

            z_nominator = np.multiply(np.square(domainKernel), z)
            if returnFirstDerivative:
                dzdx0 = np.divide(x0_dist, z_nominator)
                dzdx1 = np.divide(x1_dist, z_nominator)
                dphidx0 = dwdz*dzdx0
                dphidx1 = dwdz*dzdx1

                if returnSecondDerivative:
                    dzddx0 = np.divide((z_nominator - x0_dist*domainKernel*domainKernel*dzdx0), z_nominator*z_nominator)
                    dzddx1 = np.divide((z_nominator - x1_dist*domainKernel*domainKernel*dzdx1), z_nominator*z_nominator)
                    dzdx0dx1 = np.divide((-x0_dist*domainKernel*domainKernel*dzdx1), z_nominator*z_nominator)
            
                    dphiddx0 = dwdz*dzddx0 + dwddz* dzdx0*dzdx0
                    dphiddx1 = dwdz*dzddx1 + dwddz* dzdx1*dzdx1
                    dphidx0dx1 = dwdz*dzdx0dx1 + dwddz* dzdx0*dzdx1
            
            #construct basis vectors
            H = np.zeros(((m), self.nP))
            H[0, :] = np.ones_like(x0_dist).T
            H[1, :] = x0_dist.T
            H[2, :] = x1_dist.T
            if m > 3:
                H[3, :] = np.square(x0_dist).T
                H[4, :] = np.multiply(x0_dist,x1_dist).T
                H[5, :] = np.square(x1_dist).T


            B = H*phi.T
            H0 = np.zeros((m,1))
            H0[0] = 1
            moment = ((phi.T)* H)@(H.T)

            coef_b = np.linalg.solve(moment,H0)
            psi[i,:] = coef_b.T@B

            if returnFirstDerivative:
                dHdx0 = np.zeros(((m), self.nP))
                dHdx1 = np.zeros(((m), self.nP))
                dHdx0[1, :] = np.ones_like(x0_dist).T
                dHdx1[2, :] = np.ones_like(x0_dist).T

                if m > 3:
                    dHdx0[3, :] = 2.0*x0_dist.T
                    dHdx0[4:, :] = x1_dist.T
                    dHdx1[4, :] = x0_dist.T
                    dHdx1[5, :] = 2.0*x1_dist.T

                dBdx0 = H*dphidx0.T + dHdx0*phi.T
                dBdx1 = H*dphidx1.T + dHdx1*phi.T
                dmomentdx0 =((dphidx0.T)* H)@(H.T) \
                    + ((phi.T)* dHdx0)@(H.T) \
                    + ((phi.T)* H)@( dHdx0.T)
                dmomentdx1 =((dphidx1.T)* H)@(H.T) \
                    + ((phi.T)* dHdx1)@(H.T) \
                    + ((phi.T)* H)@( dHdx1.T)

                
                dMinvdx0 = -np.linalg.solve(moment.T, np.linalg.solve(moment, dmomentdx0).T)
                dMinvdx1  =  -np.linalg.solve(moment.T, np.linalg.solve(moment, dmomentdx1).T)
                dcoef_invMdx0H0= np.matmul(dMinvdx0,H0)
                dcoef_invMdx1H0=  np.matmul(dMinvdx1,H0)
                dpsidx0[i,:] = coef_b.T@dBdx0+ dcoef_invMdx0H0.T@B
                dpsidx1[i,:] = coef_b.T@dBdx1+ dcoef_invMdx1H0.T@B


                if returnSecondDerivative:
                    
                    dHddx0 = np.zeros(((m), self.nP))
                    dHddx1 = np.zeros(((m), self.nP))
                    dHdx0dx1 = np.zeros(((m), self.nP))
                    if m > 3:
                        dHddx0[3, :] = 2*np.ones_like(x0_dist).T
                        dHddx1[5, :] = 2*np.ones_like(x0_dist).T
                        dHdx0dx1[4, :] = np.ones_like(x0_dist).T
                
                
                    dBddx0 = H*dphiddx0.T + dHddx0*phi.T + 2*dHdx0*dphidx0.T
                    dBddx1 = H*dphiddx1.T + dHddx1*phi.T + 2*dHdx1*dphidx1.T
                    dBdx0dx1 = H*dphidx0dx1.T + dHdx0*dphidx1.T + dHdx1*dphidx0.T +dHdx0dx1*phi.T
                    dmomentddx0 =((dphiddx0.T)* H)@(H.T) \
                        + ((phi.T)* dHddx0)@(H.T) \
                        + ((phi.T)* H)@( dHddx0.T) \
                        + 2*((dphidx0.T)* dHdx0)@(H.T) \
                        + 2*((dphidx0.T)* H)@(dHdx0.T) \
                        + 2*((phi.T)*dHdx0)@( dHdx0.T)
                    dmomentddx1 =((dphiddx1.T)* H)@(H.T) \
                        + ((phi.T)* dHddx1)@(H.T) \
                        + ((phi.T)* H)@( dHddx1.T) \
                        + 2*((dphidx1.T)* dHdx1)@(H.T) \
                        + 2*((dphidx1.T)* H)@(dHdx1.T) \
                        + 2*((phi.T)*dHdx1)@( dHdx1.T)
                    dmomentdx0dx1 =((dphidx0dx1.T)* H)@(H.T) \
                        + ((dphidx0.T)* dHdx1)@(H.T) \
                        + ((dphidx0.T)* H)@( dHdx1.T) \
                        + ((dphidx1.T)* dHdx0)@(H.T) \
                        + ((phi.T)* dHdx0dx1)@(H.T) \
                        + ((phi.T)* dHdx0)@(dHdx1.T) \
                        + ((dphidx1.T)* H)@(dHdx0.T) \
                        + ((phi.T)* dHdx1)@(dHdx0.T) \
                        + ((phi.T)* H)@ (dHdx0dx1.T)

                    dMinvddx0 = (dmomentdx0.T)*(np.linalg.solve(moment, dmomentdx0).T) \
                                -np.linalg.solve(moment.T, np.linalg.solve(moment, dmomentddx0).T) \
                                -np.linalg.solve(moment.T, dmomentdx0)*(dmomentdx0.T) 
                    dMinvddx1 = (dmomentdx1.T)*(np.linalg.solve(moment, dmomentdx1).T) \
                                -np.linalg.solve(moment.T, np.linalg.solve(moment, dmomentddx1).T) \
                                -np.linalg.solve(moment.T, dmomentdx1)*(dmomentdx1.T) 
                    dMinvdx0dx1 = (dmomentdx1.T)*(np.linalg.solve(moment, dmomentdx0).T) \
                                -np.linalg.solve(moment.T, np.linalg.solve(moment, dmomentdx0dx1).T) \
                                -np.linalg.solve(moment.T, dmomentdx0)*(dmomentdx1.T) 
                    dcoef_invMddx0H0= np.matmul(dMinvddx0,H0)
                    dcoef_invMddx1H0=  np.matmul(dMinvddx1,H0)
                    dcoef_invMdx0dx1H0=  np.matmul(dMinvdx0dx1,H0)
                    dpsiddx0[i,:] = coef_b.T@dBddx0+ 2*dcoef_invMdx0H0.T@dBdx0 + dcoef_invMddx0H0.T@B
                    dpsiddx1[i,:] = coef_b.T@dBddx1+ 2*dcoef_invMdx1H0.T@dBdx1 + dcoef_invMddx1H0.T@B
                    dpsidx0dx1[i,:] = coef_b.T@dBdx0dx1+ dcoef_invMdx1H0.T@dBdx0 + dcoef_invMdx0H0.T@dBdx1 + dcoef_invMdx0dx1H0.T@B
                    
                
        if returnSecondDerivative: 
            return psi.tocsr(), dpsidx0.tocsr(), dpsidx1.tocsr(), dpsiddx0.tocsr(),  dpsiddx1.tocsr(),  dpsidx0dx1.tocsr()
        elif returnFirstDerivative:
            return psi.tocsr(), dpsidx0.tocsr(), dpsidx1.tocsr()
        else:
            return psi.tocsr()

