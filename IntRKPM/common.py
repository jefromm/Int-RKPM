
from petsc4py import PETSc
from mpi4py import MPI
from scipy import sparse
import dolfinx
from dolfinx import la, fem, io, geometry
import numpy as np
import basix
from dolfinx import cpp

def transferToForeground(u_f, u_b, M):
    """
    Transfer the solution vector from the background to the forground
    mesh.
    
    Parameters
    ----------
    u_f: Dolfin function on the foreground mesh
    u_b: PETSc vector of soln on the background mesh 
    M: extraction matrix from background to foreground.
    """
    #u_petsc = cpp.la.petsc.create_vector_wrap(u_f.x)
    u_petsc = la.create_petsc_vector_wrap(u_f.x)
    M.mult(u_b, u_petsc)
    u_f.x.scatter_forward()

def createM(V,RKPMBasis,nFields=1,returnAsSparse=False):
    '''
    generate extraction operator 

    Parameters
    ----------
    V: function space:
    RKPMBasis: basis object 
    nFields: int
    returnAsSparse: bool, option used with enrichment
    
    returns:
    M: PETSc matrix
    or 
    M_csr: scipy sparse matrix
    '''
    if nFields==1:
        dof_coords = V.tabulate_dof_coordinates()
        M_csr = RKPMBasis.evalSHP(dof_coords,returnFirstDerivative=False)
    else:
        V0, V0_to_V = V.sub(0).collapse()
        dof_coords = V0.tabulate_dof_coordinates()
        maps = [V0_to_V]
        for space in range(1,nFields):
            _, Vi_to_V = V.sub(space).collapse()
            maps += [Vi_to_V]
        M_csr_field =  RKPMBasis.evalSHP(dof_coords, returnFirstDerivative=False)
        M_lil_multiField =  sparse.lil_array(((nFields*M_csr_field.shape[0]),nFields*M_csr_field.shape[1]))
        fieldCount = 0 
        for idMap in maps: 
            M_lil_multiField[idMap,(fieldCount*M_csr_field.shape[1]):((fieldCount+1)*M_csr_field.shape[1])] = M_csr_field
            fieldCount+=1
        M_csr = M_lil_multiField.tocsr()

    if returnAsSparse:
        return M_csr
    M = PETSc.Mat().createAIJ(size=M_csr.shape,csr=(M_csr.indptr, M_csr.indices,M_csr.data))
    M.assemble()
    return M



def createMEnrichVec(V,RKPMBasis,newNp,eBasisList,enrichFuncs,nFields=1):
    if nFields > 1:
        V0, V0_to_V = V.sub(0).collapse()
        maps = [V0_to_V]
        for space in range(1,nFields):
            _, Vi_to_V = V.sub(space).collapse()
            maps += [Vi_to_V]
    else: 
        V0 = V
        
    M_preEnrich = createM(V0,RKPMBasis,nFields=1,returnAsSparse=True)
    M_lil =  sparse.lil_array(((M_preEnrich.shape[0]),(newNp)))
    #first set M_lil to og list, which will take care of any non enriched functions 
    M_lil[:,np.arange(0,RKPMBasis.nP)] = M_preEnrich
    for matID in range(eBasisList.shape[1]):
        M_lil[:,(eBasisList[:,matID][eBasisList[:,matID] >= 0 ])] = M_preEnrich[:,(eBasisList[:,matID] >= 0 )]*np.atleast_2d(enrichFuncs[matID].x.array).T
    
    if nFields == 1:
        M_csr = M_lil.tocsr()
    else:
        M_lil_multiField =  sparse.lil_array(((nFields*M_preEnrich.shape[0]),nFields*newNp))
        fieldCount = 0 
        for idMap in maps: 
            M_lil_multiField[idMap,(fieldCount*newNp):((fieldCount+1)*newNp)] = M_lil
            fieldCount+=1
        M_csr = M_lil_multiField.tocsr()

    M = PETSc.Mat().createAIJ(size=M_csr.shape,csr=(M_csr.indptr, M_csr.indices,M_csr.data))
    M.assemble()
    return M


def interpolation_matrix_nonmatching_meshes(V_1,V_0): # Function spaces from nonmatching meshes
    msh_0 = V_1.mesh
    x_0   = V_1.tabulate_dof_coordinates()
    x_1   = V_0.tabulate_dof_coordinates()

    bb_tree         = geometry.bb_tree(msh_0, msh_0.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, x_1)
    cells           = []
    points_on_proc  = []
    index_points    = []
    colliding_cells = geometry.compute_colliding_cells(msh_0, cell_candidates, x_1)

    for i, point in enumerate(x_1):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
            index_points.append(i)
            
    index_points_   = np.array(index_points)
    points_on_proc_ = np.array(points_on_proc, dtype=np.float64)
    cells_          = np.array(cells)

    ct      = dolfinx.cpp.mesh.to_string(msh_0.topology.cell_types[0])
    element = basix.create_element(basix.finite_element.string_to_family(
        "Lagrange", ct), basix.cell.string_to_type(ct), V_1.ufl_element().degree(), basix.LagrangeVariant.equispaced)

    x_ref = np.zeros((len(cells_), msh_0.topology.dim))

    for i in range(0, len(cells_)):
        geom_dofs  = msh_0.geometry.dofmap[cells_[i]]
        x_ref[i,:] = msh_0.geometry.cmaps[0].pull_back([points_on_proc_[i,:]], msh_0.geometry.x[geom_dofs])

    basis_matrix = element.tabulate(0, x_ref)[0,:,:,0]

    cell_dofs         = np.zeros((len(x_1), len(basis_matrix[0,:])))
    basis_matrix_full = np.zeros((len(x_1), len(basis_matrix[0,:])))


    for nn in range(0,len(cells_)):
        cell_dofs[index_points_[nn],:] = V_1.dofmap.cell_dofs(cells_[nn])
        basis_matrix_full[index_points_[nn],:] = basis_matrix[nn,:]

    cell_dofs_ = cell_dofs.astype(int) ###### REDUCE HERE

    I = np.zeros((len(x_1), len(x_0)))

    for i in range(0,len(x_1)):
        for j in range(0,len(basis_matrix[0,:])):
            I[i,cell_dofs_[i,j]] = basis_matrix_full[i,j]

    return I 





def outputVTX(f,V,folder,name):
    '''
    function to interpolate a ufl object onto a function space, and then
    plot the function on the function space's domain to visualize as a VTX file

    Parameters
    ----------
    f: function or expression
    V: function space
    folder: str
    name: str

    '''
    domain = V.mesh
    f_expr = fem.Expression(f, V.element.interpolation_points())
    f_fun= fem.Function(V)
    f_fun.interpolate(f_expr)
    with io.VTXWriter(domain.comm, folder+name+'.bp', [f_fun], engine="BP4") as vtx:
        vtx.write(0.0)


def outputXDMF(f,V,folder,name):
    '''
    function to interpolate a ufl object onto a function space, and then
    plot the function on the function space's domain to visualize as an xdmf file

    Typically, VTX files are easier to work with and outputVTX is prefered, but they dont 
    play well with low order discontinuous function spaces, which is when XDMF files are better

    Parameters
    ----------
    f: function or expression
    V: function space
    folder: str
    name: str
    '''
    domain = V.mesh
    f_expr = fem.Expression(f, V.element.interpolation_points())
    f_fun= fem.Function(V)
    f_fun.interpolate(f_expr)
    xdmf = io.XDMFFile(domain.comm, folder +name + ".xdmf", "w")
    xdmf.write_mesh(domain)
    xdmf.write_function(f_fun)