# -*- coding: utf-8 -*-
"""
Released on Aug 11 13:40:06 2020

@author: Giovanni Giallombardo
"""
import os

class dataInput:
    def __init__(self):
        
        self.dataSet_name    = '03'
        #   type the two-digit identifier to select the dataset according to:
        #   01 = Breast Cancer
        #   02 = Diabetes
        #   03 = Heart
        #   04 = Ionosphere
        #   05 = Brain_Tumor1
        #   06 = Brain_Tumor2
        #   07 = DLBCL
        #   08 = Leukemia/ALLAML
        
        self.kFoldSize              = 10  # Set to 10 for tenfold, 5 for fivefold, 1 for leave-one-out 
        self.hyperParamFoldSize     = 5
        
        self.methodSelector         = 1  # Set to 1 for kNORM, set to 0 for SVM
        
        self.modelSelection         = 1  # Set to 1 for best accuracy, set to 0 for balanced accuracy/sparsity
        
        self.cTilde      = 1.0
        self.mTilde      = 1.0
        
        
        self.cTildeVect  = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        
        self.pathName    = os.path.abspath('..\\dataSets') + '\\'
               
        self.nameFile    = self.pathName + self.dataSet_name
        self.fileExtension   = '.txt'
        