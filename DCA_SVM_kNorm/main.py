# -*- coding: utf-8 -*-
"""
Released on Aug 11 13:40:06 2020

@author: Giovanni Giallombardo
"""
import numpy as np
import instanceContainer as inst
import dataInput as data
import train_test_DCA as fS
import train_test_SVM as svm

if __name__ == '__main__':
    
    dataIn = data.dataInput()
    
    nameFile        = dataIn.nameFile
    fileExtension   = dataIn.fileExtension    
    
    instance = inst.instance(nameFile,fileExtension)
    
    print("n   = %i" % instance.n)
    print("l   = %i" % instance.l)
    print("m   = %i" % instance.m)
    print("l+m = %i" % instance.nSamples)
    
    print("")
    
    if dataIn.methodSelector == 0:
    
        featSVM  = svm.algorithm(instance,dataIn.kFoldSize,dataIn.hyperParamFoldSize)
        featSVM.run()
                
        print("")  
        print("Average Test   = {0:.3f}".format(np.average(featSVM.testcorrectnessVect)))
        print("Average Train  = {0:.3f}".format(np.average(featSVM.traincorrectnessVect)))
        print("ft0            = {0:.3f}".format(100.0*np.average(featSVM.nonzerosVect[:,0])/featSVM.n))
        print("ft2            = {0:.3f}".format(100.0*np.average(featSVM.nonzerosVect[:,1])/featSVM.n))
        print("ft4            = {0:.3f}".format(100.0*np.average(featSVM.nonzerosVect[:,2])/featSVM.n))
        print("ft9            = {0:.3f}".format(100.0*np.average(featSVM.nonzerosVect[:,3])/featSVM.n))
        print("Average Time   = {0:.3f}".format(np.average(featSVM.timeVect)))
       
    
    elif dataIn.methodSelector == 1:
        featSel = fS.algorithm(instance,dataIn.kFoldSize,dataIn.hyperParamFoldSize)
        featSel.run()
        
        print("")  
        print("Average Test   = {0:.3f}".format(np.average(featSel.testcorrectnessVect)))
        print("Average Train  = {0:.3f}".format(np.average(featSel.traincorrectnessVect)))
        print("ft0            = {0:.3f}".format(100.0*np.average(featSel.nonzerosVect[:,0])/featSel.n))
        print("ft2            = {0:.3f}".format(100.0*np.average(featSel.nonzerosVect[:,1])/featSel.n))
        print("ft4            = {0:.3f}".format(100.0*np.average(featSel.nonzerosVect[:,2])/featSel.n))
        print("ft9            = {0:.3f}".format(100.0*np.average(featSel.nonzerosVect[:,3])/featSel.n))
        print("Average Time   = {0:.3f}".format(np.average(featSel.timeVect)))
        