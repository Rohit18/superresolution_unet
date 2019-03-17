import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *

from torchvision.models import vgg16_bn

#Change the directories
path_hr = '/users/PAS1437/osu10674/Data/L8_512_HR'
path_lr = '/users/PAS1437/osu10674/Data/L8_64_LR_512'

il = ImageList.from_folder(path_hr)

lr = 1e-3

def calc_2_moments(tensor):
    chans = tensor.shape[1]
    tensor = tensor.view(1, chans, -1)
    n = tensor.shape[2]
    
    mu = tensor.mean(2)
    tensor = (tensor - mu[:,:,None]).squeeze(0)
    cov = torch.mm(tensor, tensor.t()) / float(n)
    
    return mu, cov

def get_style_vals(tensor):
    mean, cov = calc_2_moments(tensor)
    
    eigvals, eigvects = torch.symeig(cov, eigenvectors=True)
    
    eigroot_mat = torch.diag(torch.sqrt(eigvals.clamp(min=0)))
    
    root_cov = torch.mm(torch.mm(eigvects, eigroot_mat), eigvects.t())
    
    tr_cov = eigvals.clamp(min=0).sum()
    
    return mean, tr_cov, root_cov

def calc_l2wass_dist(mean_stl, tr_cov_stl, root_cov_stl, mean_synth, cov_synth):
    
    tr_cov_synth = torch.symeig(cov_synth, eigenvectors=True)[0].clamp(min=0).sum()
    
    mean_diff_squared = (mean_stl - mean_synth).pow(2).sum()
    
    cov_prod = torch.mm(torch.mm(root_cov_stl, cov_synth), root_cov_stl)
    
    var_overlap = torch.sqrt(torch.symeig(cov_prod, eigenvectors=True)[0].clamp(min=0)+1e-8).sum()
    
    dist = mean_diff_squared + tr_cov_stl + tr_cov_synth - 2*var_overlap
    
    return dist

def single_wass_loss(pred, targ):
    mean_test, tr_cov_test, root_cov_test = targ
    mean_synth, cov_synth = calc_2_moments(pred)
    loss = calc_l2wass_dist(mean_test, tr_cov_test, root_cov_test, mean_synth, cov_synth)
    return loss


def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr + '/' + x.name)
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data

def do_fit(save_name, epochs, lrs=slice(lr), pct_start=0.9):
    learn.fit_one_cycle(epochs, lrs, pct_start=pct_start)
    learn.save(save_name)
    learn.show_results(rows=1, imgsize=5)

if __name__ == '__main__':
    
    bs,size= 8, 512
    arch = models.resnet34
    src = ImageImageList.from_folder(path_lr).random_split_by_pct(0.05, seed=42)

    data = get_data(bs,size)

    base_loss = F.l1_loss

    vgg_m = vgg16_bn(True).features.cuda().eval()
    requires_grad(vgg_m, False)

    blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
    blocks, [vgg_m[i] for i in blocks]

    x = torch.rand([32, 256, 1024], device=0, requires_grad=True)

    feat_loss = FeatureLoss_Wass(vgg_m, blocks[2:5], [5,15,2], [3, 0.7, 0.01])

    wd = 1e-3
    learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,
                        blur=True, norm_type=NormType.Weight)
    gc.collect();

    do_fit('tmp_wassloss_64to512_largedataset', 20, slice(lr*10))

    learn.unfreeze()

    do_fit('final_wassloss_64to512_largedataset', 20, slice(5e-6,5e-4), pct_start=0.3)






