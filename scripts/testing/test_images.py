import fastai
import torchvision
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *

from torchvision.models import vgg16_bn

path_lr = '/users/PAS1437/osu10674/Data/L8_64_LR_512/'
path_hr = '/users/PAS1437/osu10674/Data/L8_512_HR/'


def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr + '/' + x.name)
           .transform(get_transforms(do_flip=True), size=size, tfm_y=True)
           .databunch(bs=bs).normalize())

    data.c = 3
    return data

if __name__ == '__main__':
    
    bs, size = 1, 512
    arch = models.resnet34
    src = ImageImageList.from_folder(path_lr).random_split_by_pct(0.1, seed=40)

    data_validate = (ImageImageList.from_folder(path_lr).random_split_by_pct(0.1, seed=42)
          .label_from_func(lambda x: path_lr + '/' + x.name)
          .transform(get_transforms(), size=size, tfm_y=True)
          .databunch(bs=1))
    data_validate.c = 3

    data = get_data(bs,size)

    size = (512,512)

    learn = unet_learner(data, arch, loss_func=F.l1_loss, blur=True, norm_type=NormType.Weight)

    learn.load('final_mse_8k');

    learn.data = data_validate

    fn = data_validate.valid_ds.x.items[1:10]

    counter = 0;

    for i in fn:

        img = open_image(i)
        print(i)

        p, img_hr, b = learn.predict(img)

        torchvision.utils.save_image(img_hr, '/users/PAS1437/osu10674/Data/mse_validation/' + str(counter) + '.png')
        
        counter = counter + 1






