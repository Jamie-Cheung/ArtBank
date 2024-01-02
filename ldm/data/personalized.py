import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
# from lavis.models import load_model_and_preprocess
import torch

imagenet_templates_smallest = [
    'a photo of a {}',
]

imagenet_templates_small = [
    '{}',
]
imagenet_dual_templates_small = [
    '{} {}'
]

per_img_token_list = ['!','@','#','$','%','^','&','(',')']

unloader = transforms.ToPILImage()  # reconvert into PIL image
import matplotlib.pyplot as plt
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(10) # pause a bit so that plots are updated

class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 specific_token = None,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 initializer_words = None,
                 ):

        self.data_root = data_root

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.num_images = len(self.image_paths)
        self._length = self.num_images 

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.specific_token = specific_token
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.initializer_words = initializer_words

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images <= len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        # self.model_blip, self.vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=torch.device("cuda"))

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images]).resize((512,512), Image.Resampling.LANCZOS)
        print(self.image_paths)
        name = self.image_paths[i % self.num_images].split('/')[-1].split('.')[0]

        if not image.mode == "RGB":
            image = image.convert("RGB")
        # _image = self.vis_processors["eval"](image).unsqueeze(0).cuda()
        # print(_image)
        # prompt_str = self.model_blip.generate({"image": _image})[0]
        # print(prompt_str)
        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        if self.per_image_tokens:
            text = random.choice(imagenet_dual_templates_small).format(placeholder_string, per_img_token_list[i % self.num_images])
        else:
            text = random.choice(imagenet_templates_small).format(placeholder_string)
            
        example["caption"] = "*"

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]
        
        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)

        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        
        return example

