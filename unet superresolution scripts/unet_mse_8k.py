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

def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
    learn.fit_one_cycle(30, lrs, pct_start=pct_start)
    learn.save(save_name)
    learn.show_results(rows=1, imgsize=5)


#get labels from func
def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr + '/' + x.name)
           .transform(get_transforms(do_flip=True), size=size, tfm_y=True)
           .databunch(bs=bs).normalize())

    data.c = 3
    return data


if __name__ == '__main__':
    
    #Define batch size and image size
    bs,size=8, 512

    #Define architecture
    arch = models.resnet34

    #Define low resolution images
    #split into training and testing sets
    src = ImageImageList.from_folder(path_lr).random_split_by_pct(0.1, seed=42)


    data = get_data(bs,size)

    #Not 100% sure about this, but definitely used for calculating loss
    wd = 1e-3
    learn = unet_learner(data, arch, wd=wd, loss_func=F.l1_loss, 
                     blur=True, norm_type=NormType.Weight)
    gc.collect();

    #First training call to action (lr defined at the beginning)
    do_fit('tmp_mse_8k', slice(lr*10))

    learn.unfreeze()
    
    do_fit('final_mse_8k', slice(1e-5,lr), pct_start=0.3)    
    
    
    