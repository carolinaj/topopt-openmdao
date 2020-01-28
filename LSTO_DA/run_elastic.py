# Single physics (elastic) discrete adjoint file
# Carolina Jauregui
# Last updated: 2020-Jan-24

from openmdao.api import Group, Problem, view_model, pyOptSparseDriver, ScipyOptimizeDriver
from openmdao.api import IndepVarComp, ExplicitComponent, ImplicitComponent
from post.plot import get_mesh, plot_solution, plot_contour

import cPickle as pickle
import numpy as np
from psutil import virtual_memory

# imports Cython wrappers for OpenLSTO_FEA, OpenLSTO_LSM
from pyBind import py_FEA
from py_lsmBind import py_LSM

# imports perturbation method (aka discrete adjoint)
from groups.PerturbGroup import *
from groups.lsm2d_SLP_Group_openlsto import LSM2D_slpGroup

# imports solvers for suboptimization
# TODO: needs to be replaced with OpenMDAO optimizer
from suboptim.solvers import Solvers
import scipy.optimize as sp_optim

saveFolder = "./save_elastic/"
import os
try:
  os.mkdir(saveFolder)
except:
  pass
try:
  os.mkdir(saveFolder + 'figs')
except:
  pass

def main(maxiter):
  print(locals())
  print("solving single physics compliance problem")

  ############################################################################
  ###########################         FEA          ###########################
  ############################################################################
  # NB: only Q4 elements + integer-spaced mesh are assumed
  nelx = 160
  nely = 80

  length_x = 160.
  length_y = 80.

  ls2fe_x = length_x/float(nelx)
  ls2fe_y = length_y/float(nely)

  num_nodes_x = nelx + 1
  num_nodes_y = nely + 1

  nELEM = nelx * nely
  nNODE = num_nodes_x * num_nodes_y

  # NB: nodes for plotting (quickfix...)
  nodes = get_mesh(num_nodes_x, num_nodes_y, nelx, nely)

  # Declare FEA object (OpenLSTO_FEA) ==============================
  fea_solver = py_FEA(lx=length_x, ly=length_y,
                      nelx=nelx, nely=nely, element_order=2)
  [node, elem, elem_dof] = fea_solver.get_mesh()

  ## validate the mesh
  if nELEM != elem.shape[0]:
    error("error found in the element")

  if nNODE != node.shape[0]:
    error("error found in the node")

  nDOF_e = nNODE * 2  # each node has two displacement DOFs

  # constitutive properties ========================================
  E = 1.
  nu = 0.3
  fea_solver.set_material(E=E, nu=nu, rho=1.0) # sets elastic material only

  # Boundary Conditions ============================================
  ## Set elastic boundary conditions
  coord_e = np.array([[0., 0.], [length_x, 0.]])
  tol_e = np.array([[1e-3, 1e3], [1e-3, 1e+3]])
  fea_solver.set_boundary(coord=coord_e, tol=tol_e)

  BCid_e = fea_solver.get_boundary()
  nDOF_e_wLag = nDOF_e + len(BCid_e)  # elasticity DOF

  # Loading Conditions =============================================
  ## Set the elastic loading conditions
  coord = np.array([length_x*0.5, 0.0])  # length_y])
  tol   = np.array([4.1, 1e-3])
  load_val = -1  # dead load
  GF_e_ = fea_solver.set_force(coord=coord, tol=tol, direction=1, f=-load_val)
  GF_e  = np.zeros(nDOF_e_wLag)
  GF_e[:nDOF_e] = GF_e_


  ############################################################################
  ###########################         LSM          ###########################
  ############################################################################
  movelimit = 0.5

  # Declare Level-set object
  lsm_solver = py_LSM(nelx=nelx, nely=nely, moveLimit=movelimit)

  # Assign holes ===================================================
  if (int(nelx)/int(nely) == 2) and (nelx >= 80):
    rad = float(nelx)/32.0 # radius of the hole
    x1  = nelx/10.     # x-coord of the center of the 1st hole 1st row
    y1  = 14.*nely/80. # y-coord of the center of the 1st row of holes
    y2  = 27.*nely/80. # y-coord of the center of the 2nd row of holes
    y3  = nely/2.      # y-coord of the center of the 3rd row of holes
    y4  = 53.*nely/80. # y-coord of the center of the 4th row of holes
    y5  = 66.*nely/80. # y-coord of the center of the 5th row of holes

    hole = array(
    	[[x1, y1, rad], [3*x1, y1, rad], [5*x1, y1, rad], [7*x1, y1, rad], [9*x1, y1, rad],
    	[2*x1, y2, rad], [4*x1, y2, rad], [6*x1, y2, rad], [8*x1, y2, rad],
    	[x1, y3, rad], [3*x1, y3, rad], [5*x1, y3, rad], [7*x1, y3, rad], [9*x1, y3, rad],
    	[2*x1, y4, rad], [4*x1, y4, rad], [6*x1, y4, rad], [8*x1, y4, rad],
    	[x1, y5, rad], [3*x1, y5, rad], [5*x1, y5, rad], [7*x1, y5, rad], [9*x1, y5, rad]])

    # NB: level set value at the corners should not be 0.0
    hole = append(hole, [[0., 0., 0.1], [0., 80., 0.1], [
    	160., 0., 0.1], [160., 80., 0.1]], axis=0)

    lsm_solver.add_holes(locx=list(hole[:, 0]), locy=list(
      hole[:, 1]), radius=list(hole[:, 2]))
  else:
    lsm_solver.add_holes([], [], [])

  lsm_solver.set_levelset()


  ############################################################################
  ########################         T.O. LOOP          ########################
  ############################################################################
  for i_HJ in range(maxiter):
    (bpts_xy, areafraction, seglength) = lsm_solver.discretise()

    # OpenMDAO ===================================================
    ## Define Group
    model = ComplianceGroup(fea_solver=fea_solver, lsm_solver=lsm_solver,
    	nelx=nelx, nely=nely, force=GF_e, movelimit=movelimit, BCid=BCid_e)

    ## Define problem for OpenMDAO object
    prob = Problem(model)

    ## Setup the problem
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'IPOPT'
    prob.driver.opt_settings['linear_solver'] = 'ma27'
    prob.setup(check=False)
    prob.run_model()

    ## Total derivative using MAUD
    total = prob.compute_totals()
    ff = total['compliance_comp.compliance', 'inputs_comp.Vn'][0]
    gg = total['weight_comp.weight', 'inputs_comp.Vn'][0]

    ## Assign object function sensitivities
    nBpts = int(bpts_xy.shape[0])
    Sf = -ff[:nBpts]                # equal to M2DO-perturbation
    Cf = np.multiply(Sf, seglength) # Shape sensitivity (integral coefficients)

    ## Assign constraint sensitivities
    Sg = -gg[:nBpts]
    Sg[Sg < - 1.5] = -1.5 # apply caps (bracketing) to constraint sensitivities
    Sg[Sg > 0.5]   =  0.5 # apply caps (bracketing) to constraint sensitivities
    Cg = np.multiply(Sg, seglength) # Shape sensitivity (integral coefficients)

    # Suboptimize ================================================
    if 1:
      suboptim = Solvers(bpts_xy=bpts_xy, Sf=Sf, Sg=Sg, Cf=Cf, Cg=Cg,
        length_x=length_x, length_y=length_y, areafraction=areafraction,
        movelimit=movelimit)
      # suboptimization
      if 1:  # simplex
        Bpt_Vel = suboptim.simplex(isprint=False)
      else:  # bisection.
        Bpt_Vel = suboptim.bisection(isprint=False)

      timestep = 1.0
      np.savetxt('a.txt',Bpt_Vel)
    elif 1: # works when Sf <- Sf / length is used (which means Cf <- actual Sf)
      bpts_sens = np.zeros((nBpts,2))
      # issue: scaling problem
      #
      bpts_sens[:,0] = Sf
      bpts_sens[:,1] = Sg

      lsm_solver.set_BptsSens(bpts_sens)
      scales = lsm_solver.get_scale_factors()
      (lb2,ub2) = lsm_solver.get_Lambda_Limits()
      constraint_distance = (0.4 * nelx * nely) - areafraction.sum()

      model = LSM2D_slpGroup(lsm_solver = lsm_solver, num_bpts = nBpts,
        ub = ub2, lb = lb2, Sf = bpts_sens[:,0], Sg = bpts_sens[:,1],
        constraintDistance = constraint_distance, movelimit=movelimit)

      subprob = Problem(model)
      subprob.setup()

      subprob.driver = ScipyOptimizeDriver()
      subprob.driver.options['optimizer'] = 'SLSQP'
      subprob.driver.options['disp'] = True
      subprob.driver.options['tol'] = 1e-10
      subprob.run_driver()

      lambdas = subprob['inputs_comp.lambdas']
      displacements_ = subprob['displacement_comp.displacements']
      # displacements_[displacements_ > movelimit] = movelimit
      # displacements_[displacements_ < -movelimit] = -movelimit

      timestep =  abs(lambdas[0]*scales[0])
      Bpt_Vel = displacements_ / timestep
      np.savetxt('a.txt',Bpt_Vel)
      # print(timestep)
      del subprob
    else: # branch: perturb-suboptim
      bpts_sens = np.zeros((nBpts,2))
      # issue: scaling problem
      #
      bpts_sens[:,0] = Sf
      bpts_sens[:,1] = Sg

      lsm_solver.set_BptsSens(bpts_sens)
      scales = lsm_solver.get_scale_factors()
      (lb2,ub2) = lsm_solver.get_Lambda_Limits()

      constraint_distance = (0.4 * nelx * nely) - areafraction.sum()
      constraintDistance = np.array([constraint_distance])
      scaled_constraintDist = lsm_solver.compute_scaledConstraintDistance(constraintDistance)

      def objF_nocallback(x):
        displacement = lsm_solver.compute_displacement(x)
        displacement_np = np.asarray(displacement)
        return lsm_solver.compute_delF(displacement_np)

      def conF_nocallback(x):
        displacement = lsm_solver.compute_displacement(x)
        displacement_np = np.asarray(displacement)
        return lsm_solver.compute_delG(displacement_np, scaled_constraintDist, 1)

      cons = ({'type': 'eq', 'fun': lambda x: conF_nocallback(x)})
      res = sp_optim.minimize(objF_nocallback, np.zeros(2), method='SLSQP', options={'disp': True},
                              bounds=((lb2[0], ub2[0]), (lb2[1], ub2[1])),
                              constraints=cons)

      lambdas = res.x
      displacements_ = lsm_solver.compute_unscaledDisplacement(lambdas)
      displacements_[displacements_ > movelimit] = movelimit
      displacements_[displacements_ < -movelimit] = -movelimit
      timestep =  1.0 #abs(lambdas[0]*scales[0])
      Bpt_Vel = displacements_ / timestep
      # scaling
      # Bpt_Vel = Bpt_Vel#/np.max(np.abs(Bpt_Vel))

    lsm_solver.advect(Bpt_Vel, timestep)
    lsm_solver.reinitialise()
    print ('loop %d is finished' % i_HJ)

    area = areafraction.sum()/(nelx*nely)
    u = prob['disp_comp.disp']
    compliance = np.dot(u, GF_e[:nDOF_e])

    # Printing/Plotting ==========================================
    if 1:  # quickplot
    	plt.figure(1)
    	plt.clf()
    	plt.scatter(bpts_xy[:, 0], bpts_xy[:, 1], 10)
    	plt.axis("equal")
    	plt.savefig(saveFolder + "figs/bpts_%d.png" % i_HJ)

    # print([compliance[0], area])
    compliance = prob['compliance_comp.compliance']
    print (compliance, area)

    fid = open(saveFolder + "log.txt", "a+")
    fid.write(str(compliance) + ", " + str(area) + "\n")
    fid.close()

    ## Saving phi
    phi = lsm_solver.get_phi()

    if i_HJ == 0:
      raw = {}
      raw['mesh'] = nodes
      raw['nodes'] = nodes
      raw['elem'] = elem
      raw['GF_e'] = GF_e
      raw['BCid_e'] = BCid_e
      raw['E'] = E
      raw['nu'] = nu
      raw['f'] = load_val
      raw['nelx'] = nelx
      raw['nely'] = nely
      raw['length_x'] = length_x
      raw['length_y'] = length_y
      raw['coord_e'] = coord_e
      raw['tol_e'] = tol_e
      filename = saveFolder + 'const.pkl'
      with open(filename, 'wb') as f:
        pickle.dump(raw, f)

    raw = {}
    raw['phi'] = phi
    filename = saveFolder + 'phi%03i.pkl' % i_HJ
    with open(filename, 'wb') as f:
      pickle.dump(raw, f)

    del model
    del prob

    mem = virtual_memory()
    print (str(mem.available/1024./1024./1024.) + "GB")
    if mem.available/1024./1024./1024. < 3.0:
      print("memory explodes at iteration %3i " % i_HJ)
      return()

if __name__ == "__main__":
  main(300)
else:
  main(1)  # testrun
