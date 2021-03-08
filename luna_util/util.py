import collections
import copy
import datetime
import gc
import time

# import torch
import numpy as np

IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])


def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    """
    Convert from Index, Row, Col to XYZ
    Args:
        coord_irc (IrcTuple):
        origin_xyz:
        vxSize_xyz:
        direction_a:
    Returns
        ret (XyzTuple): IRC coordinates converted to xyz
    """
    # flip to IRC to CRI to align with XYZ
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    return XyzTuple(*coords_xyz)


def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    """
    Convert from XYZ to Index, Row, Col
    """

    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(
        direction_a)) / vxSize_a  # inverse of the previous functions last 3 steps
    cri_a = np.round(cri_a)
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))
