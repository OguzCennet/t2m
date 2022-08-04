
import os

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

# import sys
# sys.path.insert(0, './data')
# sys.path.insert(0, './utils')
# sys.path.insert(0, './common')
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from utils.visualization import *
from utils.skeleton import Skeleton
from common.mmm import parse_motions
from common.transforms3dbatch import *
from common.quaternion import *
from renderUtils import quat2xyz
from model.model import Integrator

import torch
import pickle as pkl
import scipy.ndimage.filters as filters

import pdb



class RawData():
  def __init__(self):
    pass

  def _get_f(self):
    raise NotImplementedError
  
  def _get_df(self):
    raise NotImplementedError

  def preProcess(self):
    raise NotImplementedError
  
  def get_skeletonNpermutation(self):
    raise NotImplementedError
  
  @property
  def quat_columns(self):
    ## quaternion columns
    quat_columns = ['root_tx', 'root_ty', 'root_tz']
    for joint in self.skel.joints:
      quat_columns += ['{}_{}'.format(joint, col_suffix) for col_suffix in ['rw', 'rx', 'ry', 'rz']]

    return quat_columns

  @property
  def fke_columns(self):
    ## forward kinematics columns
    fke_columns = []
    for joint in self.skel.joints:
      fke_columns += ['{}_{}'.format(joint, col_suffix) for col_suffix in ['tx', 'ty', 'tz']]

    return fke_columns

  @property
  def pose_columns(self):
    pose_columns = []
    for joint in self.skel.joints:
      pose_columns += ['{}_{}'.format(joint, col_suffix) for col_suffix in ['rx', 'ry', 'rz']]

    return pose_columns
  
  @property
  def rifke_columns(self):
    ## Save Rotation invariant fke (rifke)
    rifke_columns = self.fke_columns + ['root_Vx', 'root_Vz', 'root_Ry', 'feet_l1', 'feet_l2', 'feet_r1', 'feet_r2']
    return rifke_columns

  @property
  def rifke_dict(self):
    raise NotImplementedError

  def output_columns(self, feats_kind):
    if feats_kind in {'euler'}:
      return self.pose_columns
    elif feats_kind in {'quaternion'}:
      return self.quat_columns
    elif feats_kind in {'fke'}:
      return self.fke_columns
    elif feats_kind in {'rifke'}:
      return self.rifke_columns

  def mat2csv(self, data, filename, columns):
    pd.DataFrame(data=data, columns=columns).to_csv(filename)
  
  def quat2fke(self, df_quat, filename_fke, filename_rifke):
    '''Save Forward Kinematics'''
    df_fke = pd.DataFrame(data=np.zeros((df_quat.shape[0], len(self.fke_columns))), columns=self.fke_columns)
    ## copying translation as is
    df_fke[['root_tx', 'root_ty', 'root_tz']] = df_quat.loc[:, ['root_tx', 'root_ty', 'root_tz']].copy()
    xyz_data = quat2xyz(df_quat, self.skel)
    df_fke.loc[:, self.fke_columns] = xyz_data.reshape(-1, np.prod(xyz_data.shape[1:]))
    #filename_fke = dir_name / Path(row[feats_kind]).relative_to(Path(path2data)/'subjects').with_suffix('.fke')
    os.makedirs(filename_fke.parent, exist_ok=True)
    df_fke.to_csv(filename_fke.as_posix())

    '''Save Rotation Invariant Forward Kinematics'''
    df_rifke = pd.DataFrame(data=np.zeros((df_quat.shape[0]-1, len(self.rifke_columns))), columns=self.rifke_columns)
    rifke_data = self.fke2rifke(xyz_data.copy())
    df_rifke[self.rifke_columns] = rifke_data[..., 3:]
    #filename_rifke = dir_name / Path(row[feats_kind]).relative_to(Path(path2data)/'subjects').with_suffix('.rifke')
    os.makedirs(filename_rifke.parent, exist_ok=True)
    df_rifke.to_csv(filename_rifke.as_posix())

    ''' Convert rifke to fke to get comparable ground truths '''
    new_df_fke = pd.DataFrame(data=self.rifke2fke(df_rifke[self.rifke_columns].values, filename_rifke).reshape(-1, len(self.fke_columns)),
                              columns=self.fke_columns)
    new_fke_dir = filename_fke.parent/'new_fke' 
    os.makedirs(new_fke_dir, exist_ok=True)
    new_df_fke.to_csv((new_fke_dir/filename_fke.name).as_posix())

    return xyz_data

  ## fke to rotation invariant fke (Holden et. al.)
  def fke2rifke(self, positions):
    """ Put on Floor """
    #fid_l, fid_r = np.array([5,6]), np.array([10,11])
    fid_l, fid_r = self.rifke_dict['fid_l'], self.rifke_dict['fid_r']
    foot_heights = np.minimum(positions[:,fid_l,1], positions[:,fid_r,1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    
    positions[:,:,1] -= floor_height

    """ Add Reference Joint """
    trajectory_filterwidth = 3
    reference = positions[:,0] * np.array([1,0,1])
    reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')    
    positions = np.concatenate([reference[:,np.newaxis], positions], axis=1)
    
    """ Get Foot Contacts """
    velfactor, heightfactor = np.array([0.05,0.05]), np.array([3.0, 2.0])
    
    feet_l_x = (positions[1:,fid_l,0] - positions[:-1,fid_l,0])**2
    feet_l_y = (positions[1:,fid_l,1] - positions[:-1,fid_l,1])**2
    feet_l_z = (positions[1:,fid_l,2] - positions[:-1,fid_l,2])**2
    feet_l_h = positions[:-1,fid_l,1]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
    
    feet_r_x = (positions[1:,fid_r,0] - positions[:-1,fid_r,0])**2
    feet_r_y = (positions[1:,fid_r,1] - positions[:-1,fid_r,1])**2
    feet_r_z = (positions[1:,fid_r,2] - positions[:-1,fid_r,2])**2
    feet_r_h = positions[:-1,fid_r,1]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
    
    """ Get Root Velocity """
    velocity = (positions[1:,0:1] - positions[:-1,0:1]).copy()
    
    """ Remove Translation """
    positions[:,:,0] = positions[:,:,0] - positions[:,0:1,0]
    positions[:,:,2] = positions[:,:,2] - positions[:,0:1,2]
    
    """ Get Forward Direction """
    #sdr_l, sdr_r, hip_l, hip_r = 19, 26, 3, 8
    sdr_l, sdr_r, hip_l, hip_r = self.rifke_dict['sdr_l'], self.rifke_dict['sdr_r'], self.rifke_dict['hip_l'], self.rifke_dict['hip_r']
    
    across1 = positions[:,hip_l] - positions[:,hip_r]
    across0 = positions[:,sdr_l] - positions[:,sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
    
    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0,1,0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')    
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    """ Remove Y Rotation """
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
    rotation = qbetween_np(forward, target)[:, np.newaxis]
    positions = qrot_np(np.repeat(rotation, positions.shape[1], axis=1), positions)
    
    """ Get Root Rotation """
    velocity = qrot_np(rotation[1:], np.repeat(velocity, rotation.shape[1], axis=1))
    rvelocity = self.get_rvelocity(rotation, forward='z', plane='xz')
    
    """ Add Velocity, RVelocity, Foot Contacts to vector """
    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)
    positions = np.concatenate([positions, velocity[:,:,0]], axis=-1)
    positions = np.concatenate([positions, velocity[:,:,2]], axis=-1)
    positions = np.concatenate([positions, rvelocity], axis=-1)
    positions = np.concatenate([positions, feet_l, feet_r], axis=-1)
    return positions

  def get_rvelocity(self, rotation, forward='z', plane='xz'):
    ## TODO - might need a reversal of inputs for qmul_np
    qs = qmul_np(rotation[1:], qinv_np(rotation[:-1]))
    ds = np.zeros(qs.shape[:-1] + (3,))
    ds[...,'xyz'.index(forward)] = 1.0
    ds = qrot_np(qs, ds)
    ys = ds[...,'xyz'.index(plane[0])]
    xs = ds[...,'xyz'.index(plane[1])]
    return np.arctan2(ys, xs)

  def rifke2fke(self, positions, filename=None):
    root_ry = torch.from_numpy(positions[..., -5]).unsqueeze(0).unsqueeze(0).float()
    pos = positions[..., :-7].reshape(positions.shape[0], -1, 3)
    pos[..., 0, [0,2]] = 0

    ''' Get Y Rotations '''
    integrator = Integrator(1, root_ry.shape[-1])
    root_ry = integrator(root_ry).squeeze(0).squeeze(0).numpy()
    rotations = np.stack([np.cos(root_ry/2), np.zeros_like(root_ry),
                          np.sin(root_ry/2), np.zeros_like(root_ry)],
                         axis=-1).astype(np.float)
    rotations = np.expand_dims(rotations, axis=1)

    ''' Rotate positions by adding Y rotations '''
    pos = qrot_np(np.repeat(qinv_np(rotations), pos.shape[1], axis=1), pos)

    ''' Rotate XZ velocity vector '''
    root_v = positions[..., -7:-5]
    root_v = np.stack([root_v[..., 0], np.zeros_like(root_v[..., 0]), root_v[..., 1]], axis=-1)
    try:
      root_v = qrot_np(qinv_np(rotations.squeeze(1)), root_v)
    except:
      pdb.set_trace()
    root_v = torch.from_numpy(root_v.transpose(1,0)).unsqueeze(0).float()

    ''' Get Root Positions from Root Velocities'''
    integrator = Integrator(3, root_v.shape[-1])
    root_t = integrator(root_v).squeeze(0).transpose(1, 0).numpy()

    ''' Add translations back to all the joints '''
    pos[..., :, 0] += root_t[..., 0:1]
    pos[..., :, 2] += root_t[..., 2:3]
    
    return pos