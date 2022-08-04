import os
import random

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
from data.raw_data import RawData

import torch
import pickle as pkl
import scipy.ndimage.filters as filters

import pdb


## permute joints to make it a DAG
def permute(parents, root=0, new_parent=-1, new_joints=[], new_parents=[]):
  new_joints.append(root)
  new_parents.append(new_parent)
  new_parent = len(new_joints) - 1
  for idx, p in enumerate(parents):
    if p == root:
      permute(parents, root=idx, new_parent=new_parent, new_joints=new_joints, new_parents=new_parents)
  return new_joints, new_parents

def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)


class KITMocap(RawData):
  def __init__(self, path2data, preProcess_flag=False):
    super(KITMocap, self).__init__()
    ## load skeleton
    self._SKELPATH = 'dataProcessing/KITMocap/skeleton.p'
    self._MMMSKELPATH = 'skeleton/mmm.xml'
    self._MMMSAMPLEPATH = 'dataProcessing/KITMocap/00001_mmm.xml'
    os.makedirs(Path(self._SKELPATH).parent, exist_ok=True)
    ## get the skeleton and permutation
    self.skel, self.permutation, self.new_joints = self.get_skeletonNpermutation()

    ## save skeleton
    pkl.dump(self.skel, open(self._SKELPATH, 'wb'))

    if preProcess_flag:
      self.preProcess(path2data)


    ## Reading data
    data = []
    for tup in os.walk(path2data):
      for filename in tup[2]:
        if Path(filename).suffix == '.xml':
          name = filename.split('_')[0]
          #print(name)
          annotpath = Path(tup[0])/(name + '_annotations.json')
          annot = json.load(open(annotpath, 'r'))
          #print('annot: ', annot)
          #print()

          exppath = Path(tup[0])/(name + '_annotations_gpt3.json') #_annotations_gpt3
          exp = json.load(open(exppath, 'r'))

          quatpath = name + '_quat.csv'
          fkepath = name + '_quat.fke'
          rifkepath = name + '_quat.rifke'
          if annot:
            for description in annot:
              data.append([(Path(tup[0])/filename).as_posix(),
                           description,
                           (Path(tup[0])/quatpath).as_posix(),
                           (Path(tup[0])/fkepath).as_posix(),
                           (Path(tup[0])/rifkepath).as_posix(),
                           random.choice(exp),
                           ])
          else:
            data.append([(Path(tup[0])/filename).as_posix(),
                         '',
                         (Path(tup[0])/quatpath).as_posix(),
                         (Path(tup[0])/fkepath).as_posix(),
                         (Path(tup[0])/rifkepath).as_posix(),
                         ''])
              
    self.df = pd.DataFrame(data=data, columns=['euler', 'descriptions', 'quaternion', 'fke', 'rifke', 'explanations'])
    #print(self.df.head(3))

    self.columns = pd.read_csv(self.df.iloc[0].quaternion, index_col=0).columns
    joints = [col[:-3] for col in self.columns]
    self.joints = []
    self.columns_dict = {}
    start = 0
    for joint in joints:
      if not self.joints:
        self.joints.append(joint)
        end = 1
      elif self.joints[-1] == joint:
        end += 1
      else:
        self.columns_dict.update({self.joints[-1]:self.columns[start:end]})
        self.joints.append(joint)
        start = end
        end = end + 1
    self.columns_dict.update({self.joints[-1]:self.columns[start:end]})

  def _get_df(self):
    return self.df

  def _get_f(self):
    return 100


  @property
  def rifke_dict(self):
    return {'fid_l':np.array([14,15]),
            'fid_r':np.array([19,20]),
            'sdr_l':6,
            'sdr_r':9,
            'hip_l':12,
            'hip_r':17}
  
  def preProcess(self, path2data):
    print('Preprocessing KIT Data')
    for tup in os.walk(path2data):
      for filename in tqdm(tup[2]):
        if Path(filename).suffix == '.xml':
          filepath = Path(tup[0])/filename
          quatpath = filename.split('_')[0] + '_quat.csv'
          quatpath = (Path(tup[0])/quatpath).as_posix()
          xyz_data, skel, joints, root_pos, rotations = self.mmm2quat(filepath)
          ## create quat dataframe
          root_pos = root_pos.squeeze(0)
          rotations = rotations.contiguous().view(rotations.shape[1], -1)
          quats = torch.cat([root_pos, rotations], dim=-1).numpy()
          columns = ['root_tx', 'root_ty', 'root_tz'] + \
                    ['{}_{}'.format(joint, axis) for joint in joints for axis in ['rw', 'rx', 'ry', 'rz']]
          df = pd.DataFrame(data=quats, columns=columns)
          df.to_csv(quatpath)
          filename_fke = Path(quatpath).with_suffix('.fke')
          filename_rifke = Path(quatpath).with_suffix('.rifke')
          self.quat2fke(df, filename_fke, filename_rifke)

  def mat2amc(self, data, filename):
    lines = ["#!OML:ASF H:",
             ":FULLY-SPECIFIED",
             ":DEGREES"]
    for count, row in enumerate(data):
      start = 0
      lines.append('{}'.format(count+1))
      for joint in self.joints:
        end = start + len(self.columns_dict[joint])
        format_str = '{} '* (len(self.columns_dict[joint]) + 1)
        format_str = format_str[:-1] ## remove the extra space
        lines.append(format_str.format(*([joint] + list(row[start:end]))))
        start = end
    lines = '\n'.join(lines) + '\n'

    os.makedirs(filename.parent, exist_ok=True)
    with open(filename,'w') as fp:
      fp.writelines(lines)

  def get_new_parents(self, parents, joints_left, joints_right, joints):
    permutation, new_parents = permute(parents)
    joints_w_root = ['root'] + joints
    new_joints = [joints_w_root[perm] for perm in permutation]
    new_joints_idx = list(range(len(new_joints)))
    new_joints_left = []
    new_joints_right = []
    for idx, jnt in enumerate(new_joints):
      if jnt[0] == 'R':
        new_joints_right.append(idx)
      else:
        new_joints_left.append(idx)
        
    return permutation, new_parents, new_joints_left, new_joints_right, new_joints

  ## KITMocap Specific
  def get_skeletonNpermutation(self):
    ## make a parents_list
    parents = [-1, 3, 0, 2, 1, 8, 9, 0, 7, 1, 6, 12, 5, 16, 17, 0, 15, 1, 14, 20, 13]
    joints_left = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    joints_right = [13, 14, 15, 16, 17, 18, 19, 20]

    ## read a demo file to get the joints
    joints, _, _, _ = self.mmm2csv(Path(self._MMMSAMPLEPATH))
    
    permutation, new_parents, new_joints_left, new_joints_right, new_joints = self.get_new_parents(parents, joints_left, joints_right, joints)

    import xml.etree.ElementTree as ET
    tree = ET.parse(self._MMMSKELPATH)
    root = tree.getroot()

    ## make an offset list
    offset_dict = self.get_offsets(root, joints)
    offset_dict.update({'root':[0,0,0]})

    new_offsets = [offset_dict[joint] for joint in new_joints]

    ## make a Skeleton
    skel = Skeleton(new_offsets, new_parents, new_joints_left, new_joints_right, new_joints)
    return skel, permutation, new_joints
    
  ## read an xml file
  def mmm2csv(self, src):
    joint_names, mmm_dict = parse_motions(src.as_posix())[0]
    root_pos = np.array(mmm_dict['root_pos'], dtype=np.float) #* 0.001 / 0.056444
    #root_pos = root_pos[:, [1,2,0]]
    root_rot = np.array(mmm_dict['root_rot'], dtype=np.float) #* 180/np.pi
    #root_rot = root_rot[:, [1,2,0]]
    joint_pos = np.array(mmm_dict['joint_pos'], dtype=np.float) #* 180/np.pi

    joint_dict = {}
    for idx, name in enumerate(joint_names):
      if name.split('_')[0][-1] != 't':
        xyz = name.split('_')[0][-1]
        joint = name.split('_')[0][:-1]
      else:
        xyz = 'y'
        joint = name.split('_')[0]
      if joint not in joint_dict:
        joint_dict[joint] = dict()
      joint_dict[joint][xyz] = joint_pos[:, idx]

    joints = []
    values = []
    for cnt, joint in enumerate(joint_dict):
      joint_vals = []
      joints.append(joint)
      for axes in ['x', 'y', 'z']:
        if axes in joint_dict[joint]:
          joint_vals.append(joint_dict[joint][axes])
        else:
          joint_vals.append(np.zeros_like(root_pos[:, 0]))
      values.append(np.stack(joint_vals, axis=1))
    values = np.stack(values, axis=0)

    return joints, root_pos, root_rot, values

  def get_offsets(self, root, Joints):
    joints = root.findall('RobotNode')
    offset_dict = {}
    for joint in joints:
      matrix = joint.findall('Transform')
      if matrix:
        offset = []
        ## switch y and z axis
        for row in ['row1', 'row3', 'row2']:
          Row = matrix[0].findall('Matrix4x4')[0].findall(row)
          offset.append(float(Row[0].attrib['c4']))
        joint_name = joint.attrib['name']
        if joint_name.split('_')[0][-6:] == 'egment':
          if joint_name[:-13] in Joints:
            offset_dict[joint_name[:-13]] = offset
        else:
          if joint_name[:-6] in Joints:
            offset_dict[joint_name[:-6]] = offset
          elif joint_name[:-7] in Joints:
            offset_dict[joint_name[:-7]] = offset
    return offset_dict

  def mmm2quat(self, path):
    joints, root_pos, root_rot, values = self.mmm2csv(path)

    ## convert to quaternions
    values_quat = euler2quatbatch(values, axes='sxyz')
    root_rot_quat = euler2quatbatch(root_rot, axes='sxyz')

    ## switch y and z axis
    ## Note the qinv_np is very important as 2 axes are being interchanged - can be proved using basic vector equations
    root_pos = root_pos[..., [0, 2, 1]] 
    values_quat = qinv_np(values_quat[..., [0, 1, 3, 2]])
    root_rot_quat = qinv_np(root_rot_quat[..., [0, 1, 3, 2]])

    rotations = np.expand_dims(np.transpose(np.concatenate((np.expand_dims(root_rot_quat, axis=0), values_quat), axis=0), axes=[1, 0, 2]), axis=0)
    root_pos = np.expand_dims(root_pos, axis=0)

    new_rotations = torch.from_numpy(rotations[:, :, self.permutation, :])
    new_root_pos = torch.from_numpy(root_pos.copy())

    xyz_data = self.skel.forward_kinematics(new_rotations, new_root_pos)[0]
    return xyz_data.numpy(), self.skel, self.new_joints, new_root_pos, new_rotations



if __name__ == '__main__':
  """PreProcessing"""
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-dataset', default='CMUMocap', type=str,
                      help='dataset kind')
  parser.add_argument('-path2data', default='../dataset/cmu-pose/all_asfamc/', type=str,
                      help='dataset kind')
  args, _ = parser.parse_known_args()
  eval(args.dataset)(args.path2data, preProcess_flag=True)
  print('Succesfully Preprocessed {} data'.format(args.dataset))
