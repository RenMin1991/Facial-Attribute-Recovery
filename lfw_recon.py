# -*- coding: utf-8 -*-
"""
reconstruction from feature embedding
RenMin 20210722
"""

import torch
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm

from recon_model import ReconModel

import pdb


# parameters
feat_extractor= 'r50'

read_folder = '/read/folder/'
write_folder = '/write/folder/'+feat_extractor+'/'


feat_path = 'features/features_'+feat_extractor+'.pth'

num_eigen_recon = 50
recon_eigen_folder = 'recon_eigen/'+feat_extractor+'/'


noise_recon_folder = 'noise_recon/'


# transformation
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


# functions
def default_loader(path):
    return Image.open(path).convert('RGB')

def scale_trans(tensor):
    tensor = (tensor + 1.) * 0.5 * 255.
    return tensor

def pth2jpg(img):
    img = img.permute(1,2,0)
    img = scale_trans(img)
    img = np.array(img)
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    return img


# recon model
recon_model = ReconModel(feat_extractor)


# reconstrction
def recon_dir(read_folder, write_folder):
    f_list = os.listdir(read_folder)
    for i in tqdm(range(len(f_list)), ncols=79):
        f = f_list[i]
    
        id_path = read_folder + f + '/'
        id_align_path = write_folder + f + '/'
    
        # create path for aligned images
        if not os.path.exists(id_align_path):
            os.makedirs(id_align_path)
            
        img_list = os.listdir(id_path)
    
        for img in img_list:
            if img[-3:]=='jpg':
                img_read_path = id_path + img
                img_write_path = id_align_path + img
                # read image and pre-process
                img =default_loader(img_read_path)
                img = transform(img)
                img = img.unsqueeze(0).cuda()
                # reconstruction
                img_recon = recon_model.recon(img).detach().cpu()
                # saving
                img_save = img_recon[0,:,:,:]
                img_save = pth2jpg(img_save)
                img_save.save(img_write_path)
        
def feat_extra_dir(read_folder):
    
    features = torch.zeros(512, 13233)
    
    img_i = 0
    
    f_list = os.listdir(read_folder)
    for i in tqdm(range(len(f_list)), ncols=79):
        f = f_list[i]
    
        id_path = read_folder + f + '/'
        img_list = os.listdir(id_path)
        for img in img_list:
            if img[-3:]=='jpg':
                img_read_path = id_path + img
                img =default_loader(img_read_path)
                img = transform(img)
                img = img.unsqueeze(0).cuda()
                # feat extraction
                feat = recon_model.feat_extract(img).detach().cpu().view(-1)
                
                features[:, img_i] = feat
                img_i = img_i + 1
                
    return features


# get eigen features
def get_eigen_feat(features):
    N = features.size(1)
    avg_feat = features.mean(1)
    features = features - avg_feat.unsqueeze(1).expand(512, N) # substract the mean vector
    C = torch.mm(features, features.t()) # covariance matrix
    eig_val, eig_feat = torch.eig(C, eigenvectors=True)
    
    return avg_feat, eig_val, eig_feat


def recon_from_feat(feat, img_write_path):
    img_recon = recon_model.decode(feat).detach().cpu()
    # saving
    img_save = img_recon[0,:,:,:]
    img_save = pth2jpg(img_save)
    img_save.save(img_write_path)
    
def recon_from_noise(write_folder, num_sam):
    for i in tqdm(range(num_sam), ncols=79):
        
        write_folder_sub = write_folder + str(i) + '/'
        
        if not os.path.exists(write_folder_sub):
            os.makedirs(write_folder_sub)
        
        write_path = write_folder_sub + str(i) + '.jpg'
        img_nosie = recon_model.sample().detach().cpu()
        # saving
        img_save = img_nosie[0,:,:,:]
        img_save = pth2jpg(img_save)
        img_save.save(write_path)
    
def recon_img(img_read_path, img_write_path):
    img =default_loader(img_read_path)
    img = transform(img)
    img = img.unsqueeze(0).cuda()
    # reconstruction
    img_recon = recon_model.recon(img).detach().cpu()
    # saving
    img_save = img_recon[0,:,:,:]
    img_save = pth2jpg(img_save)
    img_save.save(img_write_path)

if __name__ == "__main__":
    
    pdb.set_trace()

    ffhq_read_dir = '/data/renmin/dataset_face/thumbnails128x128/'
    ffhq_write_dir = 'ffhq-'+feat_extractor+'_recon/'

    for sub_d in os.listdir(ffhq_read_dir):
        print (sub_d)
        sub_dir = os.path.join(ffhq_read_dir, sub_d)
        for img_fn in os.listdir(sub_dir):
            img_read_path = os.path.join(sub_dir, img_fn)
            img_write_path = os.path.join(ffhq_write_dir, sub_d, img_fn)
            if not os.path.exists(os.path.join(ffhq_write_dir, sub_d)):
                os.makedirs(os.path.join(ffhq_write_dir, sub_d))
            recon_img(img_read_path, img_write_path)

    """
    print ('reconstruct from dir')
    print ('read folder:', read_folder)
    print ('write folder:', write_folder)
    recon_dir(read_folder, write_folder)
    
    print ('features extraction')
    features = feat_extra_dir(read_folder)
    torch.save(features, feat_path)
    
    pdb.set_trace()
    
    print ('get eigen feature')
    avg_feat, eig_val, eig_feat = get_eigen_feat(features)
    
    pdb.set_trace()
    
    #data=torch.load('recon_eigen/eigen_feat.pth')
    
    #eig_feat = data['eig_feat']
    #avg_feat = data['avg_feat']
    
    print ('reconstrcution from eigen feature')
    for i in range(num_eigen_recon):
        e_f = eig_feat[:,i].unsqueeze(0).cuda()
        e_f_plus = (eig_feat[:,i]+avg_feat).unsqueeze(0).cuda()
        
        img_write_path = recon_eigen_folder + str(i) + '.jpg'
        img_write_path_plus = recon_eigen_folder + 'plus_' + str(i) + '.jpg'
        
        recon_from_feat(e_f, img_write_path)
        recon_from_feat(e_f_plus, img_write_path_plus)
    
    
    print ('sample from noise')
    recon_from_noise(noise_recon_folder, 13233)
    """ 
        
    
    
    
    
    
            
            
