import os
import sys
import math
import datetime
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cooler
from torch.utils import data

from DDPM import *
from utils import *

parser = argparse.ArgumentParser(description='HiC2MicroC: mapping from Hi-C to Micro-C')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('-f1', '--file-hic', type=str, metavar='FILE', required=True,
                        help='a 5-kb Hi-C file in cool format')
required.add_argument('-f2', '--file-chr-sizes', type=str, metavar='FILE', required=True,
                        help='chromosome size file (chrid\tlength), Must be in order like chr1-chrX-chrY')
required.add_argument('-f3', '--file-out-prefix', type=str, metavar='FILE', required=True,
                        help='prefix for the output file')

optional.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for test (default: 64)')
optional.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA predicting')
optional.add_argument('--gpu-id', type=int, default=0, metavar='N',
                        help='GPU id (default: 0)')
optional.add_argument('--HiC-max', type=float, default=0.05, metavar='N',
                        help='the maximum value of balanced Hi-C contacts (default: 0.05)')
optional.add_argument('--chrs-avoid', type=str, default='chrY', metavar='ChrID',
                        help='the chrids to avoid (separated by comma if many) (default: chrY)')
optional.add_argument('--verbose', type=int, default=64, metavar='N',
                        help='input batch size for test (default: 64)')
args = parser.parse_args()


use_cuda = not args.no_cuda and torch.cuda.is_available()
gpu_id = args.gpu_id
device = torch.device("cuda:"+str(gpu_id) if use_cuda else "cpu")

f_hic_cool = args.file_hic
f_chr_len = args.file_chr_sizes
fout_COO = args.file_out_prefix+"_COO.gz"
fout_cool = args.file_out_prefix+".cool"
dir_out = Path(args.file_out_prefix).parent.absolute()
dir_out = str(dir_out)
best_model = dir_out + "/HiC2MicroC_5kb.pt"

batch_size = args.batch_size

chrids_avoid = args.chrs_avoid.split(",")

print("GPU:", str(gpu_id) if use_cuda else "None", flush=True)
print("Batch size:", batch_size, flush=True)
print("Chromosomes avoid:", chrids_avoid, flush=True)

### diffusion models
model_url = "http://dna.cs.miami.edu/HiC2MicroC/HiC2MicroC_5kb.pt"
device = "cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu"
model = Unet(dim=64,
             input_channels=2,
             channels=1,
             dim_mults=(1,2,4,8),
             resnet_block_groups=4,
             WeightStandardized=False,
             LinearAttn=True)
model.to(device).eval()
#model.load_state_dict(torch.load(best_model))
state_dict = torch.hub.load_state_dict_from_url(model_url, model_dir=dir_out)
model.load_state_dict(state_dict)

### parameters for diffusion
timesteps = 50
beta_start = 0.0001
beta_end = 0.95
beta = np.linspace(beta_start, beta_end, num=timesteps)
alpha = 1 - beta
alpha_cum = np.cumprod(alpha)

### parameters for 5-kb Hi-C data
res = 5000
image_size = 256
step = 50
max_bin = 456
extend_right = 5
maxV = args.HiC_max
verbose = True

### cool info
clr = cooler.Cooler(f_hic_cool)
chrids = clr.chromsizes.keys().tolist()
chr_lens = clr.chromsizes.tolist()
df_chr_len = pd.read_csv(f_chr_len, sep="\t", header=None)
chr_start_index = get_started_index_for_cooler(df_chr_len[0].tolist(), df_chr_len[1].tolist(), res=res)

time = datetime.datetime.now()
print("Start time:", time, flush=True)

for i, chrid in enumerate(chrids):

    if chrid in chrids_avoid:
        continue

    #chr_len = chr_lens[i]
    chr_len = clr.chromsizes[chrid]
    n_bins = math.ceil(chr_len / res)
    
    allInds = np.arange(0, n_bins-image_size, step)
    lastInd = allInds[len(allInds)-1]
    if (lastInd + image_size) < n_bins:
        allInds = np.append(allInds, n_bins - image_size)

    if verbose:
        print("Start predicting", chrid, chr_len, n_bins, flush=True)
    mats_hic = []
    idxes = []
    for j in allInds:
        
        idx_sj, idx_ej = j, j + image_size - 1
        psj, pej = idx_sj * res, idx_ej * res + res
        
        if pej > chr_len:
            pej = chr_len
        
        regionj = (chrid, psj, pej)
        
        for k in range(extend_right):
            
            idx_sk = j + k*step 
            idx_ek = idx_sk + image_size - 1
            psk, pek = idx_sk * res, idx_ek * res + res
            
            if idx_ek >= n_bins:
                continue
            
            if pek > chr_len:
                pek = chr_len
            
            idxes.append([idx_sj, idx_ej+1, idx_sk, idx_ek+1])
            regionk = (chrid, psk, pek)
            mat_hic = clr.matrix(balance=True).fetch(regionj, regionk)
        
            # nan to zero
            mat_hic = np.nan_to_num(mat_hic)
        
            # [> maxV] = maxV
            mat_hic[mat_hic > maxV] = maxV
        
            # first to [1, 10]. second to [0, 1] with log10
            mat_hic = np.log10((9 / maxV) * mat_hic + 1)
        
            # [0, 1] to [-1, 1]
            mat_hic = 2*mat_hic - 1
            mats_hic.append(mat_hic)

    mats_hic = np.array(mats_hic)
    mats_hic = rearrange(mats_hic, "b h w -> b 1 h w")
    dataloader_test = torch.utils.data.DataLoader(torch.from_numpy(mats_hic).type(torch.FloatTensor),
                  batch_size=batch_size, shuffle=False, drop_last=False)

    predictions = []

    with torch.no_grad():
        noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

        for step, hic in enumerate(dataloader_test):
            hic = hic.to(device)
            current_batch_size = hic.shape[0]
            img = torch.randn(hic.shape, device=device)

            for n in range(len(alpha) - 1, -1, -1):
                c1 = 1. / (alpha[n]**0.5)
                c2 = (1. - alpha[n]) / ((1 - alpha_cum[n])**0.5)

                noise_scale_n = noise_scale[n].repeat(current_batch_size)
                model_output = model(img, noise_scale_n, hic)
                img = c1 * (img - c2 * model_output)

                if n > 0:
                    noise = torch.randn_like(img)
                    sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                    img += sigma * noise
            img = torch.clamp(img, -1.0, 1.0)
            predictions.append(img.detach().cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    predictions = np.squeeze(predictions)
    predictions = np.clip(predictions, -1., 1.)
    mat = get_predict_mat(predictions, maxV, chr_len, idxes, res=res, max_bin=max_bin, step=step, image_size=image_size) 

    # COO, i, j, v
    cond1 = mat.row < mat.col
    cond2 = (mat.col - mat.row) <= max_bin
    cond3 = np.invert(np.isnan(mat.data))
    cond_all = np.logical_and(np.logical_and(cond1, cond2), cond3)

    matr = mat.row[cond_all] + chr_start_index[chrid]
    matc = mat.col[cond_all] + chr_start_index[chrid]
    matv = mat.data[cond_all]
    matv[matv > maxV] = maxV
    out = pd.DataFrame({'i': matr, 'j': matc, 'value': matv})
            
    if chrid == chrids[0]:
        out.to_csv(fout_COO, index=False, sep="\t", header=False, compression='gzip')
    else:
        out.to_csv(fout_COO, mode="a", index=False, sep="\t", header=False, compression='gzip')

    if verbose:
        print("Done. Processing", chrid, flush=True)

### COO to .cool
command = "cooler load -f coo --count-as-float --chunksize 90000000 "+f_chr_len+":"+str(res)+" " + fout_COO + " " + fout_cool
os.system(command)
        
### add weight column
if os.stat(fout_cool).st_size != 0:
    add_weight_column(fout_cool)

### clean
if os.stat(fout_COO).st_size != 0:
    os.system("rm "+fout_COO)

if verbose:
    print("DOne.", flush=True)
time = datetime.datetime.now()
print("End time:", time, flush=True)

