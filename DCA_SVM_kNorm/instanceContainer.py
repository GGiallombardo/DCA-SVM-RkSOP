# -*- coding: utf-8 -*-
"""
Released on Aug 11 13:40:06 2020

@author: Giovanni Giallombardo
"""
import re

class instance:
    def __init__(self,nameFile,fileExtension):
                
        file_dim    = open(nameFile + '_dim' + fileExtension, 'r')
        file_I      = open(nameFile + '_Indx' + fileExtension, 'r')
        file_J      = open(nameFile + '_Jndx' + fileExtension, 'r')
        file_Val    = open(nameFile + '_aVal' + fileExtension, 'r')
        file_Lab    = open(nameFile + '_Lab' + fileExtension, 'r')
        
        self.n          = int(file_dim.readline())
        self.l          = 0
        self.m          = 0
        self.nSamples   = int(file_dim.readline())
        
        numeric_const_pattern   = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
        rx                      = re.compile(numeric_const_pattern, re.VERBOSE)
        
        label_List  = file_Lab.readlines()
        iL          = file_I.readlines()
        jL          = file_J.readlines()
        valMat      = file_Val.readlines()
                
        if (self.nSamples != len(label_List)):
            print("Attention #1: something wrong in the input")
            return

        self.label      = [0 for i in range(0,self.nSamples)]
        self.valMatrix  = [[0.0 for j in range(0,self.n)] for i in range(0,len(iL))] 
        
        for i in range(0,self.nSamples):
            if label_List[i] == "0 \n" :
                self.l += 1
                self.label[i] = 0
            elif label_List[i] == "1 \n" :
                self.m +=1
                self.label[i] = 1
        
        if (self.nSamples != self.l + self.m):
            print("Attention #2: something wrong in the input")
            return
        
        for i in range(0,len(iL)):
            iInd    = rx.findall(iL[i])
            jInd    = rx.findall(jL[i])
            abVal   = rx.findall(valMat[i])
            for j in range(0,len(jInd)):
                self.valMatrix[int(iInd[j])][int(jInd[j])-1] = float(abVal[j])
        
        file_dim.close
        file_I.close
        file_J.close
        file_Val.close
        file_Lab.close

        
        