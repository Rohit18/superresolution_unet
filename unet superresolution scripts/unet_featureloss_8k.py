import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *

from torchvision.models import vgg16_bn

#Change the directories
path_hr = '/users/PAS1437/osu10674/Data/L8_512_HR'
path_lr = '/users/PAS1437/osu10674/Data/L8_64_LR_512'

il = ImageList.from_folder(path_hr)

lr = 1e-4   

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()


def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
    learn.fit_one_cycle(40, lrs, pct_start=pct_start)
    learn.save(save_name)
    learn.show_results(rows=1, imgsize=5)


#get labels from func
def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr + '/' + x.name)
           .transform(get_transforms(do_flip=True), size=size, tfm_y=True)
           .databunch(bs=bs).normalize())

    data.c = 3
    return data

def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)


if __name__ == '__main__':
    
    #Define batch size and image size
    bs,size=4,512

    #Define architecture
    arch = models.resnet34

    #Define low resolution images
    #split into training and testing sets
    src = ImageImageList.from_folder(path_lr).random_split_by_pct(0.1, seed=42)


    data = get_data(bs,size)

    #I have no idea about this and the gram matrix. Will be explained in part 2.
    t = data.valid_ds[0][1].data
    t = torch.stack([t,t])

    gram_matrix(t)

    #Defining the loss
    base_loss = F.l1_loss

    #Feature loss model for improving perceptual quality of predicted images
    vgg_m = vgg16_bn(True).features.cuda().eval()
    requires_grad(vgg_m, False)

    #Searching for the blocks just before the max pool layers to get the activation parameters
    blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
    blocks, [vgg_m[i] for i in blocks]

    #Defining the feature loss with the extracted vgg model
    feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])

    #Not 100% sure about this, but definitely used for calculating loss
    wd = 1e-3
    learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,
                     blur=True, norm_type=NormType.Weight)
    gc.collect();

    #First training call to action (lr defined at the beginning)
    do_fit('tmp_featureloss_8k', slice(lr*10))

    learn.unfreeze()

    do_fit('final_featureloss_8k', slice(1e-3,lr), pct_start=0.3)
    
    