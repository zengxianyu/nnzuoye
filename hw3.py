# coding=utf-8
import numpy as np
from PIL import Image
import random
import pdb
import matplotlib.pyplot as plt


class SOMImageCompression(object):
    def __init__(self, patch_size=3, num_code=512, num_iter=10000, alpha=0.1, topK=10):
        super(SOMImageCompression, self).__init__()
        self.patch_size, self.num_code, self.num_iter, self.alpha, self.topK = \
        patch_size, num_code, num_iter, alpha, topK
        self.weights = np.random.normal(loc=0, scale=0.01, size=(num_code, patch_size*patch_size))
        self.log = []

    def crop_img(self, img):
        patch_size = self.patch_size
        h, w = img.shape
        img = img[(h%patch_size)/2+h%patch_size%2:w-(h%patch_size)/2, (w%patch_size)/2+w%patch_size%2:w-(w%patch_size)/2]
        return img

    def split_img(self, img):
        patch_size = self.patch_size
        img = self.crop_img(img)
        img = img.astype(np.float) / 255.0
        img -= 0.5
        img *= 2
        hh, ww = img.shape
        patches = []
        for i in range(hh/patch_size):
            for j in range(ww/patch_size):
                patches += [img[patch_size*i:patch_size*i+patch_size, patch_size*j:patch_size*j+patch_size].reshape(-1)]
        return patches, hh, ww

    def train_codebook(self, train_img):
        # train
        patches, _, _ = self.split_img(train_img)
        num_iter = self.num_iter
        topK = self.topK
        alpha = self.alpha
        self.log = []
        for t in range(num_iter):
            pt = random.choice(patches)
            dis = np.sum((self.weights - pt[None, ...])**2, 1, keepdims=True)
            index = np.argsort(dis.reshape(-1), 0)[:topK]
            delta_weights = alpha*(pt-self.weights[index])
            self.weights[index] += delta_weights
            self.log += [delta_weights.sum()]

    # encode image
    def encode_img(self, img):
        patches, hh, ww = self.split_img(img)
        patches = np.array(patches)
        dis = (patches*patches).sum(1, keepdims=True) + (self.weights*self.weights).sum(1, keepdims=True).T - 2*patches.dot(self.weights.T)
        asign = np.argmin(dis, 1)
        img_code = {'code_index':asign, 'height':hh, 'width':ww}
        return img_code

    # decode image
    def decode_img(self, img_code):
        hh = img_code['height']
        ww = img_code['width']
        asign = img_code['code_index']
        recon_img = np.zeros((hh, ww))
        patch_size = self.patch_size
        ip = 0
        for i in range(hh/patch_size):
            for j in range(ww/patch_size):
                recon_img[patch_size*i:patch_size*i+patch_size, patch_size*j:patch_size*j+patch_size] = self.weights[asign[ip]].reshape(patch_size, patch_size)
                ip += 1
        recon_img = (recon_img/2+0.5)*255
        recon_img = recon_img.astype(np.uint8)
        return recon_img

    def compute_PSNR(self, img1, img2):
        mse = ((img1-img2)**2).mean()
        psnr = 10 * np.log10(255.0*255.0 / mse)
        return psnr


if __name__ == "__main__":
    train_img = Image.open('images_som/LENA.BMP')
    train_img = np.array(train_img)
    hh, ww = train_img.shape
    som = SOMImageCompression()
    som.train_codebook(train_img)

    # lena
    code = som.encode_img(train_img)
    recon_img = som.decode_img(code)
    psrn = som.compute_PSNR(som.crop_img(train_img), recon_img)
    print(u'LENA.BMP压缩比：{}，峰值信噪比：{}'.format(som.patch_size**2*8, psrn))
    recon_img = Image.fromarray(recon_img)
    recon_img.save('lena_recon.bmp')

    # CR
    test_img = Image.open('images_som/CR.BMP')
    test_img = np.array(test_img)
    code = som.encode_img(test_img)
    recon_img = som.decode_img(code)
    psrn = som.compute_PSNR(som.crop_img(test_img), recon_img)
    print(u'CR.BMP压缩比：{}，峰值信噪比：{}'.format(som.patch_size**2*8, psrn))
    recon_img = Image.fromarray(recon_img)
    recon_img.save('cr_recon.bmp')

    # HS4
    test_img = Image.open('images_som/HS4.BMP')
    test_img = np.array(test_img)
    code = som.encode_img(test_img)
    recon_img = som.decode_img(code)
    psrn = som.compute_PSNR(som.crop_img(test_img), recon_img)
    print(u'HS4.BMP压缩比：{}，峰值信噪比：{}'.format(som.patch_size**2*8, psrn))
    recon_img = Image.fromarray(recon_img)
    recon_img.save('hs4_recon.bmp')




