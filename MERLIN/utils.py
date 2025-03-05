import gc
import os
import sys
import torch

import numpy as np
from PIL import Image
from scipy import special
from scipy import signal
import matplotlib.pyplot as plt
from glob import glob
from GenerateDataset import GenerateDataset



def symetrisation_patch(ima):
    S = np.fft.fftshift(np.fft.fft2(ima[:,:,0]+1j*ima[:,:,1]))
    p = np.zeros((S.shape[0])) # azimut (ncol)
    for i in range(S.shape[0]):
        p[i] = np.mean(np.abs(S[i,:]))
    sp = p[::-1]
    c = np.real(np.fft.ifft(np.fft.fft(p)*np.conjugate(np.fft.fft(sp))))
    d1 = np.unravel_index(c.argmax(),p.shape[0])
    d1 = d1[0]
    shift_az_1 = int(round(-(d1-1)/2))%p.shape[0]+int(p.shape[0]/2)
    p2_1 = np.roll(p,shift_az_1)
    shift_az_2 = int(round(-(d1-1-p.shape[0])/2))%p.shape[0]+int(p.shape[0]/2)
    p2_2 = np.roll(p,shift_az_2)
    window = signal.gaussian(p.shape[0], std=0.2*p.shape[0])
    test_1 = np.sum(window*p2_1)
    test_2 = np.sum(window*p2_2)
    # make sure the spectrum is symetrized and zeo-Doppler centered
    if test_1>=test_2:
        p2 = p2_1
        shift_az = shift_az_1/p.shape[0]
    else:
        p2 = p2_2
        shift_az = shift_az_2/p.shape[0]
    S2 = np.roll(S,int(shift_az*p.shape[0]),axis=0)

    q = np.zeros((S.shape[1])) # range (nlin)
    for j in range(S.shape[1]):
        q[j] = np.mean(np.abs(S[:,j]))
    sq = q[::-1]
    #correlation
    cq = np.real(np.fft.ifft(np.fft.fft(q)*np.conjugate(np.fft.fft(sq))))
    d2 = np.unravel_index(cq.argmax(),q.shape[0])
    d2=d2[0]
    shift_range_1 = int(round(-(d2-1)/2))%q.shape[0]+int(q.shape[0]/2)
    q2_1 = np.roll(q,shift_range_1)
    shift_range_2 = int(round(-(d2-1-q.shape[0])/2))%q.shape[0]+int(q.shape[0]/2)
    q2_2 = np.roll(q,shift_range_2)
    window_r = signal.gaussian(q.shape[0], std=0.2*q.shape[0])
    test_1 = np.sum(window_r*q2_1)
    test_2 = np.sum(window_r*q2_2)
    if test_1>=test_2:
        q2 = q2_1
        shift_range = shift_range_1/q.shape[0]
    else:
        q2 = q2_2
        shift_range = shift_range_2/q.shape[0]


    Sf = np.roll(S2,int(shift_range*q.shape[0]),axis=1)
    ima2 = np.fft.ifft2(np.fft.ifftshift(Sf))
    return np.stack((np.real(ima2),np.imag(ima2)),axis=2)


def symetrisation_patch_test(real_part,imag_part):
    S = np.fft.fftshift(np.fft.fft2(real_part[0,:,:,0]+1j*imag_part[0,:,:,0]))
    p = np.zeros((S.shape[0])) # azimut (ncol)
    for i in range(S.shape[0]):
        p[i] = np.mean(np.abs(S[i,:]))
    sp = p[::-1]
    c = np.real(np.fft.ifft(np.fft.fft(p)*np.conjugate(np.fft.fft(sp))))
    d1 = np.unravel_index(c.argmax(),p.shape[0])
    d1 = d1[0]
    shift_az_1 = int(round(-(d1-1)/2))%p.shape[0]+int(p.shape[0]/2)
    p2_1 = np.roll(p,shift_az_1)
    shift_az_2 = int(round(-(d1-1-p.shape[0])/2))%p.shape[0]+int(p.shape[0]/2)
    p2_2 = np.roll(p,shift_az_2)
    window = signal.gaussian(p.shape[0], std=0.2*p.shape[0])
    test_1 = np.sum(window*p2_1)
    test_2 = np.sum(window*p2_2)
    # make sure the spectrum is symetrized and zeo-Doppler centered
    if test_1>=test_2:
        p2 = p2_1
        shift_az = shift_az_1/p.shape[0]
    else:
        p2 = p2_2
        shift_az = shift_az_2/p.shape[0]
    S2 = np.roll(S,int(shift_az*p.shape[0]),axis=0)

    q = np.zeros((S.shape[1])) # range (nlin)
    for j in range(S.shape[1]):
        q[j] = np.mean(np.abs(S[:,j]))
    sq = q[::-1]
    #correlation
    cq = np.real(np.fft.ifft(np.fft.fft(q)*np.conjugate(np.fft.fft(sq))))
    d2 = np.unravel_index(cq.argmax(),q.shape[0])
    d2=d2[0]
    shift_range_1 = int(round(-(d2-1)/2))%q.shape[0]+int(q.shape[0]/2)
    q2_1 = np.roll(q,shift_range_1)
    shift_range_2 = int(round(-(d2-1-q.shape[0])/2))%q.shape[0]+int(q.shape[0]/2)
    q2_2 = np.roll(q,shift_range_2)
    window_r = signal.gaussian(q.shape[0], std=0.2*q.shape[0])
    test_1 = np.sum(window_r*q2_1)
    test_2 = np.sum(window_r*q2_2)
    if test_1>=test_2:
        q2 = q2_1
        shift_range = shift_range_1/q.shape[0]
    else:
        q2 = q2_2
        shift_range = shift_range_2/q.shape[0]


    Sf = np.roll(S2,int(shift_range*q.shape[0]),axis=1)
    ima2 = np.fft.ifft2(np.fft.ifftshift(Sf))
    ima2 = ima2.reshape(1, np.size(ima2, 0), np.size(ima2, 1), 1)
    return np.real(ima2), np.imag(ima2)





def load_train_data(filepath, patch_size, batch_size, stride_size, n_data_augmentation): #TODO: add control on training data: exit if does not exists
    datagen = GenerateDataset()
    imgs = datagen.generate_patches(src_dir=filepath, pat_size=patch_size, step=0,
                             stride=stride_size, bat_size=batch_size, data_aug_times=n_data_augmentation)
    return imgs

def load_sar_images(filelist):
    if not isinstance(filelist, list):
        im = np.load(filelist)
        im = im[:,:,:2]
        return np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), 2)
    data = []
    for file in filelist:
        im = np.load(file)
        im = im[:,:,:2]
        data.append(np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), 2))
    return data





def store_data_and_plot(im, threshold, filename):
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    im = Image.fromarray(im.astype('float64')).convert('L')
    im.save(filename.replace('npy','png'))


def save_sar_images(denoised, noisy, imagename, save_dir, groundtruth=None):
    # choices = {'marais1':190.92, 'marais2': 168.49, 'saclay':470.92, 'lely':235.90, 'ramb':167.22,
    #        'risoul':306.94, 'limagne':178.43, 'saintgervais':560, 'Serreponcon': 450.0,
    #       'Sendai':600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
    #       'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0, 'domancy': 560, 'Brazil': 103.0, 'masguyot': 4931.0, 'corrected':269.0}
    threshold = None
    # for x in choices:
    #     if x in imagename:
    #         threshold = choices.get(x)
    if threshold is None: threshold = np.mean(noisy) + 3 * np.std(noisy)

    if groundtruth:
        groundtruthfilename = save_dir+"/groundtruth_"+imagename
        np.save(groundtruthfilename,groundtruth)
        store_data_and_plot(groundtruth, threshold, groundtruthfilename)

    denoisedfilename = save_dir + "/denoised_" + imagename
    np.save(denoisedfilename, denoised)
    store_data_and_plot(denoised, threshold, denoisedfilename)

   # noisyfilename = save_dir + "/noisy_" + imagename
   # np.save(noisyfilename, noisy)
   # store_data_and_plot(noisy, threshold, noisyfilename)


def save_real_imag_images(real_part, imag_part, imagename, save_dir):
    choices = {'marais1': 190.92, 'marais2': 168.49, 'saclay': 470.92, 'lely': 235.90, 'ramb': 167.22,
               'risoul': 306.94, 'limagne': 178.43, 'saintgervais': 560, 'Serreponcon': 450.0,
               'Sendai': 600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
               'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0, 'Brazil': 103.0, 'masguyot': 4931.0, 'corrected':269.0}
    threshold = None
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)
    if threshold is None: threshold = np.mean(imag_part) + 3 * np.std(imag_part)

    realfilename = save_dir + "/denoised_real_" + imagename
    np.save(realfilename, real_part)
    store_data_and_plot(real_part, threshold, realfilename)

    imagfilename = save_dir + "/denoised_imag_" + imagename
    np.save(imagfilename, imag_part)
    store_data_and_plot(imag_part, threshold, imagfilename)

def save_real_imag_images_noisy(real_part, imag_part, imagename, save_dir):
    choices = {'marais1': 190.92, 'marais2': 168.49, 'saclay': 470.92, 'lely': 235.90, 'ramb': 167.22,
               'risoul': 306.94, 'limagne': 178.43, 'saintgervais': 560, 'Serreponcon': 450.0,
               'Sendai': 600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
               'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0, 'Brazil': 103.0, 'masguyot': 4931.0, 'corrected':269.0}
    threshold = None
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)
    if threshold is None: threshold = np.mean(np.abs(imag_part)) + 3 * np.std(np.abs(imag_part))

    realfilename = save_dir + "/noisy_real_" + imagename
    np.save(realfilename, real_part)
    store_data_and_plot(np.sqrt(2)*np.abs(real_part), threshold, realfilename)

    imagfilename = save_dir + "/noisy_imag_" + imagename
    np.save(imagfilename, imag_part)
    store_data_and_plot(np.sqrt(2)*np.abs(imag_part), threshold, imagfilename)

def save_segmented_image(out_probs, imagename,sample_dir):
    segmentation_filename = sample_dir+"/Segmentation_"+imagename
    np.save(segmentation_filename,out_probs)
    threshold_seg = 0.5
    out_probs[out_probs<threshold_seg] = 0
    out_probs[out_probs>=threshold_seg] = 255
    store_data_and_plot(out_probs,255,segmentation_filename)

def save_groundtruth_samples(y, l, imagename,sample_dir):
    threshold = 8000.0
    store_data_and_plot(y,threshold,sample_dir+"/SAR_"+imagename)
    store_data_and_plot(255*l,255,sample_dir+"/Groundtruth_"+imagename)


def save_residual(filepath, residual):
    residual_image = np.squeeze(residual);
    plt.imsave(filepath, residual_image)


def cal_psnr(Shat, S):
    # takes amplitudes in input
    # Shat: a SAR amplitude image
    # S:    a reference SAR image
    P = np.quantile(S, 0.99)
    res = 10 * np.log10((P ** 2) / np.mean(np.abs(Shat - S) ** 2))
    return res


def tf_psnr(im1, im2):
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))


def init_weights(m):
      if type(m) == torch.nn.Linear:
          torch.nn.init.xavier_normal_(m.weight,gain=2.0)
          print('[*] inizialized weights')


def evaluate(model, loader):
    outputs = [model.validation_step(batch) for batch in loader]
    outputs = torch.tensor(outputs).T
    loss, accuracy = torch.mean(outputs, dim=1)
    return {"loss": loss.item(), "accuracy": accuracy.item()}


def save_model(model,destination_folder):
    """
      save the ".pth" model in destination_folder
    """
    # Check whether the specified path exists or not
    isExist = os.path.exists(destination_folder)

    if not isExist:

      # Create a new directory because it does not exist
      os.makedirs(destination_folder)
      print("The new directory is created!")

      torch.save(model.state_dict(),destination_folder+"/model.pth")

    else:
      torch.save(model.state_dict(),destination_folder+"/model.pth")

def save_checkpoint(model,destination_folder,epoch_num,optimizer,loss):
    """
      save the ".pth" checkpoint in destination_folder
    """
    # Check whether the specified path exists or not
    isExist = os.path.exists(destination_folder)

    if not isExist: os.makedirs(destination_folder)

    torch.save({
            'epoch_num': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, destination_folder+"/checkpoint_"+str(epoch_num)+".pth")
    print("\n Checkpoint saved at :",destination_folder)
