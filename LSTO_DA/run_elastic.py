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

saveFolder = "elastic"
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

  ##############################################################################
  ##########################         FEA          ##############################
  ##############################################################################
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

  # Declare FEA object (OpenLSTO_FEA) ================================
  fea_solver = py_FEA(lx=length_x, ly=length_y,
                      nelx=nelx, nely=nely, element_order=2)
  [node, elem, elem_dof] = fea_solver.get_mesh()

  # validate the mesh
  if nELEM != elem.shape[0]:
      error("error found in the element")
  if nNODE != node.shape[0]:
      error("error found in the node")

  # constitutive properties ==========================================
  E = 1.
  nu = 0.3
  fea_solver.set_material(E=E, nu=nu, rho=1.0) # sets elastic material only

  # Boundary Conditions ==============================================
  ## Set elastic boundary conditions
  coord_e = np.array([[0., 0.], [length_x, 0.]])
  tol_e = np.array([[1e-3, 1e3], [1e-3, 1e+3]])
  fea_solver.set_boundary(coord=coord_e, tol=tol_e)

  BCid_e = fea_solver.get_boundary()
  nDOF_e_wLag = nDOF_e + len(BCid_e)  # elasticity DOF

  # Loading Conditions ===============================================
  ## Set the elastic loading conditions
  coord = np.array([length_x*0.5, 0.0])  # length_y])
  tol   = np.array([4.1, 1e-3])
  load_magnitude = -1  # dead load
  GF_e_ = fea_solver.set_force(coord=coord, tol=tol, direction=1, f=-f)
  GF_e  = np.zeros(nDOF_e_wLag)
  GF_e[:nDOF_e] = GF_e_

  ##############################################################################
  ##########################         LSM          ##############################
  ##############################################################################
  movelimit = 0.5

  # Declare Level-set object
  lsm_solver = py_LSM(nelx=nelx, nely=nely, moveLimit=movelimit)

  # Assign holes =====================================================
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