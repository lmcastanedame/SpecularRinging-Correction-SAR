import time
import numpy as np
import os

from utils import *
from scipy import special
import argparse

import torch
import numpy as np



class AE(torch.nn.Module):

    def __init__(self,batch_size,eval_batch_size,device):
        super().__init__()

        self.batch_size=batch_size
        self.eval_batch_size=eval_batch_size
        self.device=device

        self.x = None
        self.height = None
        self.width = None
        self.out_channels = None
        self.kernel_size_cv2d = None
        self.stride_cv2d = None
        self.padding_cv2d = None
        self.kernel_size_mp2d = None
        self.stride_mp2d = None
        self.padding_mp2d = None
        self.alpha = None
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.leaky = torch.nn.LeakyReLU(0.1)

        self.enc0 = torch.nn.Conv2d(in_channels=1, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc1 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc2 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc3 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc4 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc5 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc6 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')

        self.dec5 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec5b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec4 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec4b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec3 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec3b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec2 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec2b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec1a = torch.nn.Conv2d(in_channels=97, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec1b = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec1 = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')

        self.upscale2d = torch.nn.UpsamplingNearest2d(scale_factor=2)


    def forward(self,x ,batch_size):
        """  Defines a class for an autoencoder algorithm for an object (image) x

        An autoencoder is a specific type of feedforward neural networks where the
        input is the same as the
        output. It compresses the input into a lower-dimensional code and then
        reconstruct the output from this representattion. It is a dimensionality
        reduction algorithm

        Parameters
        ----------
        x : np.array
        a numpy array containing image

        Returns
        ----------
        x-n : np.array
        a numpy array containing the denoised image i.e the image itself minus the noise

        """
        #x=torch.reshape(x, [batch_size, 1, 256, 256])
        skips = [x]

        n = x

        # ENCODER
        n = self.leaky(self.enc0(n))
        n = self.leaky(self.enc1(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc2(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc3(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc4(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc5(n))
        n = self.pool(n)
        n = self.leaky(self.enc6(n))


        # DECODER
        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec5(n))
        n = self.leaky(self.dec5b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec4(n))
        n = self.leaky(self.dec4b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec3(n))
        n = self.leaky(self.dec3b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec2(n))
        n = self.leaky(self.dec2b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec1a(n))
        n = self.leaky(self.dec1b(n))

        n = self.dec1(n)

        return x-n

    def loss_function(self,output,target,batch_size, eps_loss, eps_log):
      """ Defines and runs the loss function

      Parameters
      ----------
      output :
      target :
      batch_size :

      Returns
      ----------
      loss: float
          The value of loss given your output, target and batch_size

      """
      # ----- loss -----
      log_hat_R = 2*output 
      b_log = torch.log(torch.square(target) + eps_log)/2
      loss = torch.mean(0.5*log_hat_R + torch.exp(2*b_log - log_hat_R))
      return loss

    def training_step(self, batch,batch_number, eps_loss, eps_log):

      """ Train the model with the training set

      Parameters
      ----------
      batch : a subset of the training date
      batch_number : ID identifying the batch

      Returns
      -------
      loss : float
        The value of loss given the batch

      """
      x, y = batch
      x=x.to(self.device)
      y=y.to(self.device)

      if (batch_number%2==0):
        x=0.5*torch.log(torch.square(x)+eps_log)
        out = self.forward(x,self.batch_size)
        loss = self.loss_function(out, y,self.batch_size, eps_loss, eps_log)

      else:
        y=0.5*torch.log(torch.square(y)+eps_log)
        out = self.forward(y,self.batch_size)
        loss = self.loss_function(out,x,self.batch_size, eps_loss, eps_log)

      return loss

    def validation_step(self, batch,image_num,epoch_num,eval_files,eval_set,sample_dir, eps_loss, eps_log):
      ###rajouter étape si image pas multiple de 32
      '''
      Renvoie l'image en amplitude débruitée
      '''

      image_real_part,image_imaginary_part = batch

      image_real_part=image_real_part.to(self.device)
      image_imaginary_part=image_imaginary_part.to(self.device)

      image_real_part_log =torch.log(torch.square(image_real_part) + eps_log)/2
      image_imaginary_part_log = torch.log(torch.square(image_imaginary_part) + eps_log)/2

      out_real = self.forward(image_real_part_log,self.eval_batch_size)
      out_imaginary = self.forward(image_imaginary_part_log,self.eval_batch_size)

      loss_real = self.loss_function(out_real, image_imaginary_part,self.eval_batch_size, eps_loss, eps_log)
      loss_imaginary = self.loss_function(out_imaginary, image_real_part,self.eval_batch_size, eps_loss, eps_log)

      loss = (loss_real + loss_imaginary)/2

      real_part = np.exp((np.squeeze(out_real.cpu().numpy())))
      imaginary_part = np.exp((np.squeeze(out_imaginary.cpu().numpy())))

      output_clean_image = 0.5*(np.square(real_part)+np.square(imaginary_part)) #0.5
      # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

      noisyimage = np.sqrt(np.squeeze(np.square(image_real_part.cpu().numpy())+np.square(image_imaginary_part.cpu().numpy())))
      outputimage = np.sqrt(np.squeeze(output_clean_image))


      print('Denoised image %d'%(image_num))
      print('min image', np.min(outputimage))
      # rename and save
      imagename = eval_files[image_num].replace(eval_set, "")
      imagename = imagename.replace('.npy', '_epoch_' + str(epoch_num) + '.npy')

      save_sar_images(outputimage, noisyimage, imagename,sample_dir)

      return loss.cpu().detach().numpy()
    


    def test_step1(self, im, image_num, test_files, test_set, test_dir):

        image_real_part,image_imaginary_part = im
        
        image_real_part=image_real_part.to(self.device)
        image_imaginary_part=image_imaginary_part.to(self.device)

        # Normalization
        image_real_part_log =torch.log(torch.square(image_real_part) + 0.001)/2
        image_imaginary_part_log = torch.log(torch.square(image_imaginary_part) + 0.001)/2

        out_real = self.forward(image_real_part_log,self.eval_batch_size)
        out_imaginary = self.forward(image_imaginary_part_log,self.eval_batch_size)

        real_part = np.exp((np.squeeze(out_real.cpu().numpy())))
        imaginary_part = np.exp((np.squeeze(out_imaginary.cpu().numpy())))

        output_clean_image = 0.5*(np.square(real_part)+np.square(imaginary_part)) 

        imagename = test_files[image_num].replace(test_set, "")

        print('Denoised image %d'%(image_num))

        noisy = image_real_part**2+image_imaginary_part**2
        save_sar_images(output_clean_image, np.squeeze(np.asarray(noisy.cpu().numpy())), imagename, test_dir)


    def test_step(self, im, image_num, test_files, test_set, test_dir):
      '''
      Renvoie l'image en amplitude débruitée
      '''

      image_real_part, image_imaginary_part = im

      image_real_part = image_real_part.to(self.device)
      image_imaginary_part = image_imaginary_part.to(self.device)

      # Taille de l'image
      h, w = image_real_part.shape[-2:]
      patch_size = 512  
      overlap = 32  
      step = patch_size - overlap

      def process_patch(patch_real, patch_imag):
          patch_real_log = torch.log(torch.square(patch_real) + 0.001) / 2
          patch_imag_log = torch.log(torch.square(patch_imag) + 0.001) / 2

          out_real = self.forward(patch_real_log, self.eval_batch_size)
          out_imaginary = self.forward(patch_imag_log, self.eval_batch_size)

          real_part = np.exp(np.squeeze(out_real.cpu().numpy()))
          imaginary_part = np.exp(np.squeeze(out_imaginary.cpu().numpy()))

          return 0.5 * (np.square(real_part) + np.square(imaginary_part))

      if h > 5000 or w > 5000:
          output_clean_image = np.zeros((h, w))
          weight_map = np.zeros((h, w)) 
          for i in range(0, h, step):
              for j in range(0, w, step):
                  i_end = min(i + patch_size, h)
                  j_end = min(j + patch_size, w)

                  patch_real = image_real_part[..., i:i_end, j:j_end]
                  patch_imag = image_imaginary_part[..., i:i_end, j:j_end]

                  clean_patch = process_patch(patch_real, patch_imag)

                  output_clean_image[i:i_end, j:j_end] += clean_patch[:i_end - i, :j_end - j]
                  weight_map[i:i_end, j:j_end] += 1  

          output_clean_image /= np.maximum(weight_map, 1)

      else:
          output_clean_image = process_patch(image_real_part, image_imaginary_part)

      output_clean_image = np.sqrt(output_clean_image)
      imagename = test_files[image_num].replace(test_set, "")
      print(f'Denoised image {image_num}')

      noisy = image_real_part**2 + image_imaginary_part**2
      save_sar_images(output_clean_image, np.squeeze(np.asarray(np.sqrt(noisy.cpu().numpy()))), imagename, test_dir)
  
        


    
