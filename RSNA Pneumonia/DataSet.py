#Importing the Libraries
import os   #used for listing directories in a Folder, or Items in a File
import torch  # Pytorch Neural Network Package
import pandas as pd    # To import csv Files
import numpy as np      # To operate on Numpy Arrays
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader   # To create a custom DataSet in Pytorch
from torchvision import transforms, utils     # To operate on Datasets such as Normilazation
import pydicom    # To deal Operate on DICOM Images

#Let us a create a class for the Training Dataset

class PnomniaData_Training(Dataset):
    
    # Lets create a PnemoniaDataset Object Creator
    def __init__(self , csv_file , root_dir , transform ):
        # Let us give the property of the oject a csv File
        self.data = pd.read_csv(csv_file)
        #Let us set the Root Directory
        self.root_dir = root_dir
        self.transform = transform
        
        
    # Let us Find the Size of the DataSet
    def __len__(self):
        return len(self.data)-1
    
    #Lets Create a method to Get an Item of the Dataset
    def __getitem__(self, idx):  #Idx is the index of the wanted Item of The Dataset
        # using the iloc  Property let us get an Item of the Csv
        patientData = self.data.iloc[idx]
        patiendId = patientData['patientId'] #let us get a patient id from the PatientId Colounm
        #let us create a coloumn matrix(its size is 5), this matrix contains x,y,width,height,target, we do so by concatinating them
        x = patientData['x']  #let us get the x coodinate  from the "x" Colounm
        y = patientData['y']
        width = patientData['width']
        height = patientData['height']
        target = patientData['Target']
        XandY = np.concatenate((x,y) ,axis = None)
        XandYandWidth = np.concatenate((XandY,width), axis = None)
        XandYandWidthAndHight = np.concatenate((XandYandWidth,height), axis = None)

        
        #With the Patient Id concatinated with the root director, lets go the Photos folder to get a DICOM Image
        dcm_file = str(self.root_dir)+ patiendId +".dcm"
        dcm_data = pydicom.read_file(dcm_file)
        #let us get the pixels of the DICOM Image
        image = dcm_data.pixel_array
        
        #Let us create a dictionary. This dictionary has an Id, the x,y.. coodicates and the image pixels of the DICOM Image)
        sample = { 'XandYandWidthAndHight':XandYandWidthAndHight, 'image':image,'target':target}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    

#A class to convert to Our Data to Pytorch Tensors
class ToTensor_Training(object):
    
    def __call__(self, sample):
        XandYandWidthAndHight, image, target = sample['XandYandWidthAndHight'], sample['image'], sample['target']
        XandYandWidthAndHight = np.array([XandYandWidthAndHight])
        XandYandWidthAndHight=XandYandWidthAndHight.reshape(-1,1)
        XandYandWidthAndHight = torch.from_numpy(XandYandWidthAndHight)
        XandYandWidthAndHight = XandYandWidthAndHight.type(torch.FloatTensor)

        target = np.array([target])
        target = torch.from_numpy(target)
        target = target.type(torch.FloatTensor)
        image = image[..., np.newaxis]
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)
        return { 'XandYandWidthAndHight':XandYandWidthAndHight, 'image':image,'target':target}
    
    
    
    
 #Let us a create a class for the Testing Dataset
class PnomniaData_Testing(Dataset):
    
    def __init__(self , root_dir , transform ):
        self.root_dir = root_dir
        self.transform = transform
        
        
    def __len__(self):
        self.listOfFiles = os.listdir(self.root_dir)
        return len(self.listOfFiles)
    
    def __getitem__(self, idx):
        patientIdWithExtention = self.listOfFiles[idx]
        patientId = patientIdWithExtention[:-4]
        dcm_file = str(self.root_dir) + patientIdWithExtention
        dcm_data = pydicom.read_file(dcm_file)
        image = dcm_data.pixel_array
        
        sample = {'patientId':patiendId, 'image':image}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
class ToTensor_Testing(object):
    
    def __call__(self, sample):
        patientId , image = sample['patientId'], sample['image']
        image = image[..., np.newaxis]
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        
        return  {'patientId':patientId, 'image':image}



if __name__ == '__main__':
 
	trainingDataset_instance = PnomniaData_Training(csv_file='/Users/abdulsalamyazid/Desktop/Projects/RSNA Pneumonia/Dataset/train_label.csv',
	                             root_dir = '/Users/abdulsalamyazid/Desktop/Projects/RSNA Pneumonia/Dataset/train_images/',
	                             transform = ToTensor_Training())


	train_loader = DataLoader(trainingDataset_instance , batch_size = 4,
                        shuffle = True ,num_workers = 4)

	sample_batch = next(iter(train_loader))
	print(sample_batch['XandYandWidthAndHight'].shape)