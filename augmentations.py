import torch
import numpy as np
from corruptor import *
import sys

def embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset=False):
    device = x_cont.device
    x_categ = x_categ + model.categories_offset.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ)
    n1,n2 = x_cont.shape
    _, n3 = x_categ.shape
    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(n1,n2, model.dim)
        for i in range(model.num_continuous):
            x_cont_enc[:,i,:] = model.simple_MLP[i](x_cont[:,i])
    else:
        raise Exception('This case should not work!')    


    x_cont_enc = x_cont_enc.to(device)
    cat_mask_temp = cat_mask + model.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)

    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    # replacing feature encoding where mask==0
    # Q: what are mask? Why do need it?
    # Ans: 
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    if vision_dset:
        
        pos = np.tile(np.arange(x_categ.shape[-1]),(x_categ.shape[0],1))
        pos =  torch.from_numpy(pos).to(device)
        pos_enc =model.pos_encodings(pos)
        x_categ_enc+=pos_enc

    return x_categ, x_categ_enc, x_cont_enc




def mixup_data(x1, x2 , lam=1.0, y= None, use_cuda=True):
    '''Returns mixed inputs, pairs of targets'''

    batch_size = x1.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)


    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    if y is not None:
        y_a, y_b = y, y[index]
        return mixed_x1, mixed_x2, y_a, y_b
    
    return mixed_x1, mixed_x2


def add_noise(x_categ,x_cont, noise_params = {'noise_type' : ['cutmix'],'lambda' : 0.1,}, mr=0, mt='mcar', corruptor1= None, opt = None):
    lam = noise_params['lambda']
    device = x_categ.device
    batch_size = x_categ.size()[0]
    # print('Missing rate is XXX ', mr)
    # print('noise param', noise_params)
    if 'cutmix' in noise_params['noise_type']:
        df_categ = pd.DataFrame(x_categ.cpu().numpy())
        df_cont = pd.DataFrame(x_cont.cpu().numpy())
        num_unique_values = len(torch.unique(x_categ))
        if num_unique_values > 1:
            # print('Inside unique')
            # Concatenate categorical and continuous dataframes
            # df_combined = pd.concat([df_categ, df_cont], axis=1)
            # Initialize corruptor
            '''corruptor_settings ={
                'method': 'mcar_missing',
                'corruption_rate': 0.6,
                'missing': mr,
                'missing_type': mt,
                'mice': 'LinearRegression'
            }
            corruptor = Corruptor(df_cont, corruptor_settings)

            # Get missing data and mask
            # _, df_missing_combined = corruptor._mcar_missing(df_combined)
            # mask = corruptor.mask
            # torch.tensor(df_combined.values, dtype=torch.float32)
            data, mask = corruptor(torch.tensor(df_cont.values))
            # print('nan count', torch.sum(torch.isnan(data)))
            corruptor_settings ={
                'method': 'draw',
                'corruption_rate': 0.6,
                'missing': mr,
                'missing_type': mt,
                'mice': 'LinearRegression'
            }
            corruptor1 = Corruptor(data, corruptor_settings)'''

            data = corruptor1(torch.tensor(data.values))
            
            # print(data.type)
            data = pd.DataFrame(data)
            index = torch.randperm(batch_size)
            cat_corr_mask = torch.from_numpy(np.random.choice(2,(x_categ.shape),p=[lam,1-lam])).to(device) #Binomial Mask for Cat
            x1 =  x_categ[index,:]
            x_categ_corr = x_categ.clone().detach()
            x_categ_corr[cat_corr_mask==0] = x1[cat_corr_mask==0]
            df_missing_categ = x_categ_corr
            df_missing_cont = data
            x_categ_missing = df_missing_categ
            x_cont_missing = torch.from_numpy(df_missing_cont.to_numpy()).to(device)

            return x_categ_missing, x_cont_missing
        else:
            if opt.corruption_type == 'cutmix':
                data, _ = corruptor1(x_cont)
                x_categ_missing = x_categ
                x_cont_missing = data
            elif opt.corruption_type == 'zeroes':
                data, mask = corruptor1(x_cont)
                x_categ_missing = x_categ
                x_cont_missing = data * (1 - mask)  
            elif opt.corruption_type == 'no_corruption':    
                x_categ_missing = x_categ
                x_cont_missing = x_cont
      
            return x_categ_missing, x_cont_missing
        # return x_categ_corr, x_cont_corr
    elif noise_params['noise_type'] == 'missing':
        x_categ_mask = np.random.choice(2,(x_categ.shape),p=[lam,1-lam])
        x_cont_mask = np.random.choice(2,(x_cont.shape),p=[lam,1-lam])
        x_categ_mask = torch.from_numpy(x_categ_mask).to(device)
        x_cont_mask = torch.from_numpy(x_cont_mask).to(device)
        return torch.mul(x_categ,x_categ_mask), torch.mul(x_cont,x_cont_mask)
    
    else:
        print("yet to write this")
