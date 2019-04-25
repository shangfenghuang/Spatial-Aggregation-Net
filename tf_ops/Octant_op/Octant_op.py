"""
Author: Jiang Mingyang
email: jmydurant@sjtu.edu.cn
pointSIFT module op, do not modify it !!!
"""

import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

Octant_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_Octant_so.so'))

def Octant_select(xyz, xyz_new, radius):
    """
    :param xyz: (b, n, 3) float
    :param radius: float
    :return: (b, n, 8) int
    """
    idx = Octant_module.cube_select(xyz, xyz_new, radius)
    return idx


ops.NoGradient('CubeSelect')

def Octant_select_two(xyz, xyz_new, radius):
    """
    :param xyz: (b, n, 3) float
    :param radius:  float
    :return: idx: (b, n, 16) int
    """
    idx = Octant_module.cube_select_two(xyz, xyz_new, radius)
    return idx


ops.NoGradient('CubeSelectTwo')

def Octant_select_four(xyz, xyz_new, radius):
    """
    :param xyz: (b, n, 3) float
    :param radius:  float
    :return: idx: (b, n, 32) int
    """
    idx = Octant_module.cube_select_four(xyz, xyz_new, radius)
    return idx


ops.NoGradient('CubeSelectFour')

def Octant_select_eight(xyz, xyz_new, radius):
    """
    :param xyz: (b, n, 3) float
    :param radius:  float
    :return: idx: (b, n, 32) int
    """
    idx = Octant_module.cube_select_eight(xyz, xyz_new, radius)
    return idx


ops.NoGradient('CubeSelectEight')