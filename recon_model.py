from models import pspEncoderv2, resnet50, Generator, IDDEncoder, latentDiscrimonator
import torch
import torch.nn.functional as F
from face_utils.arcface import MobiFaceNet, Backbone, remove_module_dict, l2_norm
import math





class ReconModel(object):
    def __init__(self, reg_model):
        # reg_model: 'mb' or 'r50'

        # Mean Latent
        self.mean_latent = torch.load("mean_wplus.pth").cuda()
        self.mean_latent.requires_grad = False
        
        # delta encoder
        #self.encoder = pspEncoderv2(resnet50(False, omega_only=True), num_latents=[3, 4, 7], depth=50, omega_only=True).cuda()
        #ckpt_e = "ckpts/delta_256/delta_gc_000018.pth"
        #self.encoder.load_state_dict(torch.load(ckpt_e), strict=True)
        #self.encoder.eval()

        # ID Encoder
        self.id_encoder = IDDEncoder(14, 8).cuda()
        ckpt_e = "ckpts/delta_256/id_delta_gc_rm_"+reg_model+".pth"
        self.id_encoder.load_state_dict(torch.load(ckpt_e), strict=True)
        self.id_encoder.eval()

        # Generator
        ckpt_gn = "checkpoint/550000.pt"
        self.generator = Generator(256, 512, 8, channel_multiplier=2).cuda()
        self.generator.load_state_dict(torch.load(ckpt_gn, map_location=lambda storage, loc: storage)["g_ema"], strict=False)
        self.generator.eval()
            

        # Feature Extractor
        if reg_model=='mb':
            self.face_model = MobiFaceNet().cuda()
            self.face_model.load_state_dict(remove_module_dict(torch.load("checkpoint/arcface_mb.pth")))
        elif reg_model=='r50':
            self.face_model = Backbone(50).cuda() 
            self.face_model.load_state_dict(remove_module_dict(torch.load("checkpoint/arcface_r50.pth")))
        self.face_model.eval()
    
    def recon(self, x):
        with torch.no_grad():
            x = F.interpolate(x, (112, 112), mode='bilinear', align_corners=True)
            feat = self.face_model(x)
            id_lats = self.id_encoder(feat)
            id_lats = self.mean_latent + id_lats
            
            x_recon, _ = self.generator([id_lats, None])
            
        return x_recon
    
    #def delta_recon(self, x):
        #with torch.no_grad():
            #x = F.interpolate(x, (256, 256), mode='bilinear', align_corners=True)
            #lats, _ = self.encoder(x)
            #lats = self.mean_latent + lats
            
            #x_recon, _ = self.generator([lats, None])
            
        #return x_recon
    
    def decode(self, feat):
        with torch.no_grad():
            id_lats = self.id_encoder(feat)
            id_lats = self.mean_latent + id_lats
            
            x_recon, _ = self.generator([id_lats, None])
            
        return x_recon
    
    def feat_extract(self, x):
        with torch.no_grad():
            x = F.interpolate(x, (112, 112), mode='bilinear', align_corners=True)
            feat = self.face_model(x)
            
        return feat
    
    def sample(self):
        D = 512
        
        #angles = 2.*math.pi*torch.rand(D-1) - math.pi
        #feat_sample = torch.zeros(D)
        
        #for i in range(D-1):
            #feat_sample[i] = torch.sin(angles[i]) * torch.prod(torch.cos(angles[:i]))
        #feat_sample[D-1] = torch.prod(torch.cos(angles))
        
        #sig = 0
        #while sig==0:
            #feat_sample = 0.0823*(2. * torch.rand(1, D) - 1.)
            #scale_feat = ((feat_sample**2).sum())**0.5
            #if scale_feat <= 1.:
                #sig = 1
        
        feat_sample = torch.randn(D)
        scale_feat = ((feat_sample**2).sum())**0.5
        feat_sample = feat_sample / scale_feat
        
        feat_sample = feat_sample.unsqueeze(0).cuda()
        with torch.no_grad():
            id_lats = self.id_encoder(feat_sample)
            id_lats = self.mean_latent + id_lats
            
            x_recon, _ = self.generator([id_lats, None])
            
        return x_recon


if __name__ == "__main__":
    recon_model = ReconModel('mb')
    x = torch.rand(1,3,128,128).cuda()
    x_recon = recon_model.recon(x)
    print (x_recon.size())
    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.enable = True
