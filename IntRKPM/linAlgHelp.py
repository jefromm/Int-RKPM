
from petsc4py import PETSc 
from scipy import sparse


def  assembleLinearSystemBackground(a_f, L_f, M):
    """
    Assemble the linear system on the background mesh, with
    variational forms defined on the foreground mesh.
    
    Parameters
    ----------
    a_f: LHS PETSc matrix
    L_f: RHS PETSc matrix
    M: extraction petsc matrix 
    
    Returns
    -------  
    A_b: PETSc matrix on the background mesh
    b_b: PETSc vector on the background mesh
    """

    A_b = AT_R_A(M, a_f)
    b_b = AT_x(M, L_f)
    return A_b, b_b

def AT_R_A(A, R):
    """
    Compute "A^T*R*A". A,R are "petsc4py.PETSc.Mat".

    Parameters
    -----------
    A : petsc4py.PETSc.Mat
    R : petsc4py.PETSc.Mat

    Returns
    ------ 
    ATRA : petsc4py.PETSc.Mat
    """
    AT = A.transpose()
    ATR = AT.matMult(R)
    ATT = A.transpose()
    ATRA = ATR.matMult(ATT)
    return ATRA

def AT_x(A, x):
    """
    Compute b = A^T*x.
    Parameters
    ----------
    A : petsc4py.PETSc.Mat
    x : petsc4py.PETSc.Vec
    Returns
    -------
    b_PETSc : petsc4py.PETSc.Vec
    """
    
    b_PETSc = A.createVecRight()
    A.multTranspose(x, b_PETSc)
    return b_PETSc


def np2PMat(a):
    """
    Converts from numpy array to petsc matrix via scipy sparse matrix
    Parameters
    ----------
    a: np array
    Returns
    -------
    petsc_mat : petsc4py.PETSc.Mat
    """
    csr_mat = sparse.csr_matrix(a)
    petsc_mat = PETSc.Mat().createAIJ(size=csr_mat.shape,csr=(csr_mat.indptr, csr_mat.indices,csr_mat.data))
    return petsc_mat

def np2PVec(a):
    """
    Converts np.vector to petsc vector 
    Parameters
    ----------
    a: np array
    Returns
    -------
    petsc_mat : petsc4py.PETSc.Mat
    """
    petsc_vec = PETSc.Vec().createWithArray(a)
    return petsc_vec

def p2npMat(a):
    """
    Converts from petsc matrix to numpy array via scipy sparse matrix
    Parameters
    ----------
    a:petsc4py.PETSc.Mat
    Returns
    -------
    petsc matrix 
    """
    return a.getValuesCSR()

def p2npVec(a):
    """
    Converts petsc vector to np.vector
    Parameters
    ----------
    a: petsc vec
    Returns
    -------
    np array
    """
    return a.getArray()


def solveKSP(A,b,u,method='gmres', PC='jacobi',
            rtol=1E-8, atol=1E-9, max_it=1000000,monitor=True,gmr_res=3000):
    """
    solve linear system A*u=b
    A: PETSC Matrix
    b: PETSC Vector
    u: PETSC Vector

    For guidance on what these options might mean I (JF) refer users (and my future self) to 
        the following page:
        https://scicomp.stackexchange.com/questions/513/why-is-my-iterative-linear-solver-not-converging

    options:
    method: str, 'mumps' (not actually a KSP), 'gmres', 'gcr', 'cg'
    PC: str, preconditions, N/A if method is mumps, 'jacobi', 'ASM', 'ICC',  'ILU'
    rtol:flt, relative tolerance
    atol:flt, absolute tolerance
    max_it:int, maximum iterations
    monitor:bool
    gmr_res: int, gmres restart param
    """

    if method == None:
        method='gmres'
    if PC == None:
        PC='jacobi'

    if method == 'mumps':
        # note- mumps is a direct solver, and may be slower than iterative solvers, 
        # but is more likely to provide a solution as it doesnt need to converge
        ksp = PETSc.KSP().create() 
        ksp.setTolerances(rtol=rtol, atol = atol, max_it= max_it)
        opts = PETSc.Options("mat_mumps_")
        # icntl_24: controls the detection of “null pivot rows", 1 for on, 0 for off
        opts["icntl_24"] = 1
        # cntl_3: is used to determine null pivot rows
        opts["cntl_3"] = 1e-12           

        A.assemble()
        ksp.setOperators(A)
        ksp.setType('preonly')
        pc=ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')
        ksp.setUp()

        ksp.solve(b,u)
        return 


    ksp = PETSc.KSP().create() 
    ksp.setTolerances(rtol=rtol, atol = atol, max_it= max_it)

    if method == 'gmres': 
        ksp.setType(PETSc.KSP.Type.FGMRES)
    elif method == 'gcr':
        ksp.setType(PETSc.KSP.Type.GCR)
    elif method == 'cg':
        ksp.setType(PETSc.KSP.Type.CG)


    if PC == 'jacobi':
        A.assemble()
        ksp.setOperators(A)
        pc = ksp.getPC()
        pc.setType("jacobi")
        ksp.setUp()
        ksp.setGMRESRestart(300)

    elif PC == 'ASM':
        A.assemble()
        ksp.setOperators(A)
        ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setType("asm")
        pc.setASMOverlap(1)
        ksp.setUp()
        localKSP = pc.getASMSubKSP()[0]
        localKSP.setType(PETSc.KSP.Type.FGMRES)
        localKSP.getPC().setType("lu")
        ksp.setGMRESRestart(gmr_res)

    elif PC== 'ICC':
        A.assemble()
        ksp.setOperators(A)
        ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setType("icc")
        ksp.setUp()
        ksp.setGMRESRestart(gmr_res)

    elif PC== 'ILU':
        A.assemble()
        ksp.setOperators(A)
        ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setType("hypre")
        pc.setHYPREType("euclid")
        ksp.setUp()
        ksp.setGMRESRestart(gmr_res)

    elif PC == 'ILUT':
        A.assemble()
        ksp.setOperators(A)
        ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setType("hypre")
        pc.setHYPREType("pilut")
        ksp.setUp()
        ksp.setGMRESRestart(gmr_res)

    print('Ready to solve')
   
    opts = PETSc.Options()
    opts["ksp_monitor"] = None
    opts["ksp_view"] = None
    ksp.setFromOptions()
    ksp.solve(b,u)
    print('Solving')
    
    history = ksp.getConvergenceHistory()
    if monitor:
        print('Converged in', ksp.getIterationNumber(), 'iterations.')
        print('Convergence history:', history)
