#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from glob import glob
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Dataset import *
from model import *
from utils import *
from fpdf import FPDF

basedir = './train1'
datasetdir = './dataset'

torch.manual_seed(2)
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=20, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='# images in batch')
parser.add_argument('--val_batch_size', dest='val_batch_size', type=int, default=1, help='# images in batch')

parser.add_argument('--eps_loss', dest='eps_loss', type=float, default=1e-3, help='# eps used in the loss computation')
parser.add_argument('--eps_log', dest='eps_log', type=float, default=1e-3, help='# eps used to compute the log images')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=256, help='# size of a patch')
parser.add_argument('--stride_size', dest='stride_size', type=int, default=128, help='# size of the stride')
parser.add_argument('--n_data_augmentation', dest='n_data_augmentation', type=int, default=1, help='# data aug techniques')
parser.add_argument('--lr', dest='lr', type=float, default = 1e-5, help='initial learning rate for adam')
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.001, help='weight decay for adam')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default=basedir+"/saved_model", help='models are saved here')

parser.add_argument('--sample_dir', dest='sample_dir', default=basedir+"/sample", help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default=basedir+"/test/", help='test sample are saved here')
parser.add_argument('--eval_set', dest='eval_set', default=datasetdir+'/validation/', help='dataset for eval in training')
parser.add_argument('--test_set', dest='test_set', default=datasetdir+'/test/', help='dataset for testing')
parser.add_argument('--training_set', dest='training_set', default=datasetdir+'/train/', help='dataset for training')
parser.add_argument('--device', dest='device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), help='gpu or cpu')

args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)

import matplotlib.pyplot as plt

def generate_training_report(history, filename=basedir+"/training_report.pdf"):
    class PDF(FPDF):
      def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Training Report', 0, 1, 'C')

      def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

      def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    pdf = PDF()
    pdf.add_page()
    

    average_loss = (np.mean(history["train_loss"]) + np.mean(history["validation_loss"])) / 2

    margin = 0.3
    min_loss = min(min(history["train_loss"]), min(history["validation_loss"]))
    max_loss = max(max(history["train_loss"]), max(history["validation_loss"]))
    lower_limit = max(min_loss, average_loss - margin)
    upper_limit = min(max_loss, average_loss + margin)

    pdf.chapter_title("Training and Validation Loss")
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["validation_loss"], label="Validation Loss")
    plt.ylim(lower_limit, upper_limit)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    loss_plot_path = os.path.join(basedir, "loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()

    pdf.image(loss_plot_path, x=10, y=None, w=190)

    pdf.chapter_title("Summary")
    summary = f"Total epochs: {len(history['train_loss'])}\n"
    summary += f"Final training loss: {history['train_loss'][-1]:.4f}\n"
    summary += f"Final validation loss: {history['validation_loss'][-1]:.4f}\n"
    pdf.chapter_body(summary)

    pdf.chapter_title("Training Parameters")
    params = f"Epochs: {args.epoch}\n"
    params += f"Batch size: {args.batch_size}\n"
    params += f"Validation batch size: {args.val_batch_size}\n"
    params += f"Patch size: {args.patch_size}\n"
    params += f"Stride size: {args.stride_size}\n"
    params += f"Data augmentation techniques: {args.n_data_augmentation}\n"
    params += f"Learning rate: {args.lr}\n"
    params += f"Weight decay: {args.weight_decay}\n"
    params += f"Use GPU: {args.use_gpu}\n"
    params += f"Phase: {args.phase}\n"
    params += f"Checkpoint directory: {args.ckpt_dir}\n"
    params += f"Sample directory: {args.sample_dir}\n"
    params += f"Test directory: {args.test_dir}\n"
    params += f"Evaluation set: {args.eval_set}\n"
    params += f"Test set: {args.test_set}\n"
    params += f"Training set: {args.training_set}\n"
    params += f"Device: {args.device}\n"
    params += f"Eps_log: {args.eps_log}\n"
    params += f"Eps_loss: {args.eps_loss}\n"

    # Ajouter les fichiers contenus dans les répertoires
    def list_files(directory, label):
      try:
          files = os.listdir(directory)
          return f"{label}:\n" + "\n".join(f"  - {file}" for file in files) + "\n"
      except FileNotFoundError:
          return f"{label}: Directory not found\n"
      except NotADirectoryError:
          return f"{label}: Not a directory\n"

    params += list_files(args.test_set, "Files in Test Directory")
    params += list_files(args.eval_set, "Files in Evaluation Set")
    params += list_files(args.training_set, "Files in Training Set")

    pdf.chapter_body(params)

    pdf.output(filename)
    print(f"Training report saved as {filename}")

def fit(model,train_loader,val_loader,epochs,lr_list,gn_list,eval_files,eval_set,checkpoint_folder):
  """ Fit the model according to the given evaluation data and parameters.

  Parameters
  ----------
  model : model as defined in main
  train_loader : Pytorch's DataLoader of training data
  val_loader : Pytorch's DataLoader of validation data
  lr_list : list of learning rates
  eval_files : .npy files used for evaluation in training
  eval_set : directory of dataset used for evaluation in training

  Returns
  ----------
  self : object
    Fitted estimator.

  """


  train_losses = []
  val_losses=[]
  history={}

  ckpt_files = glob(checkpoint_folder+"/checkpoint_*")
  if len(ckpt_files)==0:
      epoch_num = 0
      model.apply(init_weights)
      optimizer = torch.optim.Adam(model.parameters(), lr=lr_list[epoch_num])
      loss = 0.0
      print("[*] Not find pre-trained model! Start training froms scratch")

  else:
    max_file = max(ckpt_files, key=os.path.getctime)
    checkpoint = torch.load(max_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    epoch_num = checkpoint['epoch_num'] 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_list[min(epoch_num, len(lr_list) - 1)])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    print("[*] Model restored! Resume training from latest checkpoint at "+max_file)


  scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
  start_time = time.time()

  for epoch in range(epoch_num,epochs):
      epoch_num += 1
      print("\nEpoch", epoch + 1)
      print("Scheduler learning rate:", lr_list[epoch])
      print("Gradient norm:", gn_list[epoch])
      print("***************** \n")
      optimizer = torch.optim.Adam(model.parameters(), lr=lr_list[epoch])

      #Train
      model.train()
      loss_mean = 0
      for i, batch in enumerate(train_loader, 0):

            optimizer.zero_grad()
            loss = model.training_step(batch,i, args.eps_loss, args.eps_log)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gn_list[epoch])
            optimizer.step()
            
            loss_mean+=loss.item()
            print(f"[{epoch + 1}, {i + 1}] time: {time.time() - start_time:.4f}, loss: {loss:.3f}")
            
      loss_mean = loss_mean/len(train_loader)
      train_losses.append(loss_mean)
      print('For epoch', epoch+1,'the mean training loss is :',loss_mean)

      save_checkpoint(model, checkpoint_folder, epoch + 1, optimizer, loss_mean)

      with torch.no_grad():
        image_num=0
        val_loss_mean = 0
        for batch in val_loader:
            val_loss=model.validation_step(batch,image_num,epoch_num,eval_files,eval_set,args.sample_dir, args.eps_loss, args.eps_log)
            val_loss = val_loss.item()
            print(val_loss)
            image_num=image_num+1
            val_loss_mean+=val_loss

        val_loss_mean /= len(val_loader)
        scheduler.step(val_loss_mean)
        val_losses.append(val_loss_mean)

      print('For epoch', epoch+1,'the last validation loss is :',val_loss_mean)

  history["train_loss"]=train_losses
  history["validation_loss"]=val_losses

  generate_training_report(history)

  return history



def denoiser_train(model,lr_list,gn_list):
  """ Runs the denoiser algorithm for the training and evaluation dataset

  Parameters
  ----------
  model : model as defined in main
  lr_list : list of learning rates

  Returns
  ----------
  history : list of both training and validation loss

  """
  print("début  de l'entraînement")
  # Prepare train DataLoader
  train_data = load_train_data(args.training_set, args.patch_size, args.batch_size, args.stride_size, args.n_data_augmentation) # range [0; 1]
  print(train_data.shape)
  train_dataset = Dataset(train_data)

  train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,drop_last=True)
  print("chargement du dataset") 
  # Prepare Validation DataLoader
  eval_dataset = ValDataset(args.eval_set) # range [0; 1]
  eval_loader = torch.utils.data.DataLoader(eval_dataset,batch_size=args.val_batch_size,shuffle=False,drop_last=True)
  eval_files = glob(args.eval_set+'*.npy')

  # Train the model
  history =fit(model,train_loader,eval_loader,args.epoch,lr_list,gn_list,eval_files,args.eval_set,args.ckpt_dir)

  # Save the model
  save_model(model,args.ckpt_dir)
  print("\n model saved at :",args.ckpt_dir)
  return history

def denoiser_test(model):
    """ Runs the test denoiser algorithm

    Parameters
    ----------
    denoiser : model as defined in main

    Returns
    ----------

    """
    # Prepare Validation DataLoader
    test_dataset = ValDataset(args.test_set) # range [0; 1]
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.val_batch_size,shuffle=False,drop_last=True)
    test_files = glob(args.test_set+'*.npy')
    print(test_files)


    val_losses=[]
    ckpt_files = glob(args.ckpt_dir+"/checkpoint_*")
    if len(ckpt_files)==0:
        print("[*] Not find pre-trained model! ")
        return None

    else:
        max_file = max(ckpt_files, key=os.path.getctime)
        checkpoint = torch.load(max_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        #model.train()

        print("[*] Model restored! Start testing...")

        with torch.no_grad():
            image_num=0
            for batch in test_loader:
              print(image_num)
              print(test_files)
              model.test_step(batch,image_num, test_files,args.test_set,args.test_dir)
              image_num=image_num+1


def main():
  if not os.path.exists(args.ckpt_dir):
      os.makedirs(args.ckpt_dir)
  if not os.path.exists(args.sample_dir):
      os.makedirs(args.sample_dir)
  if not os.path.exists(args.test_dir):
      os.makedirs(args.test_dir)
  # learning rate list
  lr = args.lr * np.ones([args.epoch])
  lr[4:20] = lr[0]/10
  lr[20:] = lr[0]/100
  # gradient norm list
  gn = 1.0*np.ones([args.epoch])



  model = AE(args.batch_size,args.val_batch_size,args.device)
  model.to(args.device)

  if args.phase == 'train':
      denoiser_train(model,lr,gn)
  elif args.phase == 'test':
      denoiser_test(model)
  else:
      print('[!]Unknown phase')
      exit(0)


if __name__ == '__main__':
    main()
