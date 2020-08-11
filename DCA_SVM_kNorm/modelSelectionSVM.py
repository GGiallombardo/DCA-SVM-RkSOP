# -*- coding: utf-8 -*-
"""
Released on Aug 11 13:40:06 2020

@author: Giovanni Giallombardo
"""
import numpy as np
import cplex as cpx
import dataInput as data
import copy
from   sklearn.model_selection import StratifiedKFold

class modelSelect:
    def __init__(self,instance,abMatrix,label,cVector,foldSize):
        self.n              = instance.n
        self.l              = instance.l
        self.m              = instance.m
        self.nSamples       = instance.nSamples
        self.ABMatrix       = np.array(abMatrix)
        self.label          = np.array(label)
        self.cTildeVector   = cVector
        self.dataIn         = data.dataInput()
        self.objectiveStar  = np.inf
        self.foldSize       = foldSize   
        self.cVector        = cVector
        
    def run(self):
        kF = StratifiedKFold(n_splits=self.foldSize,shuffle=True, random_state = 1)
        self.testcorrectnessVect  = np.zeros(self.foldSize)
        self.traincorrectnessVect = np.zeros(self.foldSize)
        self.zerosVect            = np.zeros(self.foldSize)
        self.nonzerosVect         = np.zeros((self.foldSize,4))
        self.threshold = [1.0, pow(10,-2), pow(10,-4), pow(10,-9)]
        self.avgCorrectness = []
        self.avgZeros       = []
        
        cIndex = 0
        for self.cTilde in self.cTildeVector:
            index = 0
            for train, test in kF.split(self.ABMatrix,self.label):
                abTrain     = np.array(self.ABMatrix[train])
                abTest      = np.array(self.ABMatrix[test])
                labTrain    = np.array(self.label[train])
                labTest     = np.array(self.label[test])
                self.train(abTrain,labTrain,abTest,labTest,index)
                index += 1
            self.avgCorrectness.append(np.average(self.testcorrectnessVect))
            self.avgZeros.append(np.average(self.zerosVect))
            cIndex += 1
        if self.dataIn.modelSelection == 1 :
            occurrences = np.zeros(self.foldSize)
            occurrences = np.where(self.avgCorrectness == np.max(self.avgCorrectness),1,0)
            cIndMax = np.argmax(occurrences)
        elif self.dataIn.modelSelection == 0 :
            cIndMax = self.balancedModelSelection()
        return self.cTildeVector[cIndMax]
            
    def train(self,abTrain,labTrain,abTest,labTest,foldIndex):
        self.setSize(labTrain)
        self.variablesMaker(abTrain,labTrain)
        self.constraintsMaker(abTrain,labTrain)
        self.lpSolver()
        self.getSolution()
        for j in range(4):
            self.nonzerosVect.itemset((foldIndex,j), self.findZeroNorm(self.wTry,self.threshold[j]))
        self.nonzerosWCounter = 0
        for i in range(self.n):
            if np.abs(self.wTry[i]) > self.threshold[3]:
                self.nonzerosWCounter += 1
        self.zerosWCounter = 100 * (self.n - self.nonzerosWCounter)/self.n 
        self.testcorrectnessVect[foldIndex] = self.correctnessCheck(self.wTry,self.gammaTry,abTest,labTest)
        self.traincorrectnessVect[foldIndex] = self.correctnessCheck(self.wTry,self.gammaTry,abTrain,labTrain)
        self.zerosVect[foldIndex] = self.zerosWCounter
    
    def balancedModelSelection(self):
        indMax = np.argmax(self.avgCorrectness)
        diffCorrectness = np.zeros(len(self.cVector))
        diffZeros       = np.zeros(len(self.cVector))
        diffZeros       = self.avgZeros - self.avgZeros[indMax]
        diffCorrectness = self.avgCorrectness[indMax] - self.avgCorrectness 
        diffZeros[indMax] = 0.0
        diffCorrectness[indMax] = 0.0
        for i in range((len(self.cVector))):
            if diffZeros[i] < 0.0:
                diffZeros[i] = 0.0
                diffCorrectness[i] = 0.0
        ratio = np.zeros((len(self.cVector)))
        for i in range((len(self.cVector))):
            if diffCorrectness[i] > 0.0:
                ratio[i] = diffZeros[i]/diffCorrectness[i]
        if np.max(ratio) > 1.0:
            return np.argmax(ratio)
        else:
            return indMax
    
    def lpSolver(self):
        self.p.set_log_stream(None) 
        self.p.set_results_stream(None)
        self.p.set_warning_stream(None) 
        self.p.set_error_stream(None)
        self.p.solve()
        
    def getSolution(self):
        self.objValue       = self.p.solution.get_objective_value()
        self.csi            = self.p.solution.get_values(self.csicolName)
        self.zeta           = self.p.solution.get_values(self.zetacolName)
        self.wPlusTry       = np.array(self.p.solution.get_values(self.wpluscolName))
        self.wMinusTry      = np.array(self.p.solution.get_values(self.wminuscolName))
        self.wTry           = (self.wPlusTry-self.wMinusTry).tolist()
        self.gammaTry       = self.p.solution.get_values(self.gammacolName)
         
    def functionEvaluation(self,w,gamma,ab,lab):
        self.findKNorm(w)
        self.aError = 0.0
        self.bError = 0.0
        self.objectiveError = 0.0
        self.objectiveTry   = 0.0
        for i in range(self.lSize+self.mSize):
            if lab[i] == 0:
                self.aError += max(0.0, np.dot(ab[i],w) - gamma + 1.0)
            else:
                self.bError += max(0.0, -np.dot(ab[i],w) + gamma + 1.0)
        
        self.objectiveError += self.cTilde*(self.aError + self.bError)
        self.objectiveTry   += self.objectiveError
        self.objectiveTry   += self.n*self.kNorm[self.n-1]
        kN = np.array(self.kNorm)
        self.objectiveTry   -= np.sum(kN)    
        
    def optimalityCheck(self):
        if self.objectiveTry < self.objectiveStar:
            return False
        else:
            return True
    
    def correctnessCheck(self, w, gamma,ab,lab):
        aCorrect = 0
        bCorrect = 0
        countA   = 0
        countB   = 0
        for i in range(len(lab)):
            if lab[i] == 0:
                countA += 1
                if np.dot(ab[i],w) <= gamma:
                    aCorrect += 1                    
            else:
                countB += 1
                if np.dot(ab[i],w) >= gamma:
                    bCorrect += 1
        correctness = aCorrect + bCorrect
        pcgCorrectness = 100.0 * (correctness / (countA+countB))
        return pcgCorrectness
        
    def solutionUpdate(self):
        self.wStar = copy.deepcopy(self.wTry)
        self.gammaStar = self.gammaTry
        self.objectiveStar = self.objectiveTry
        
    def findKNorm(self,w):
        absVector       = np.absolute(w)
        tempVector      = np.array(absVector)
        sumTemp = 0.0
        self.kNorm   = [0.0 for i in range(self.n)]
        self.kIndMax = [0.0 for i in range(self.n)]
        for i in range(self.n):
            self.kIndMax[i]              = np.argmax(tempVector)
            sumTemp                     += absVector[self.kIndMax[i]]
            tempVector[self.kIndMax[i]]  = np.NINF
            self.kNorm[i]                = sumTemp
        
    def findZeroNorm(self,w,epsilon):
        counter = 0
        for i in range(len(w)):
            if abs(w[i]) > epsilon:
                counter += 1
        return counter
        
    def setSize(self,lab):
        self.lSize = 0
        self.mSize = 0
        
        for i in range(len(lab)):
            if lab[i] == 0:
                self.lSize += 1
            else:
                self.mSize += 1
    
    
    def variablesMaker(self,ab,lab):
        self.objectiveCoeff     = []
        self.upperBounds        = []
        self.lowerBounds        = []
        self.colName            = []
        self.csicolName         = []
        self.zetacolName        = []
        self.wpluscolName       = []
        self.wminuscolName      = []
        self.gammacolName       = []
        
        
        for i in range(self.lSize):
            self.objectiveCoeff.append(self.cTilde)
            self.upperBounds.append(cpx.infinity)
            self.lowerBounds.append(0.0)
            self.colName.append("csi["+str(i+1)+"]")
            self.csicolName.append("csi["+str(i+1)+"]")
        
        for i in range(self.mSize):
            self.objectiveCoeff.append(self.cTilde)
            self.upperBounds.append(cpx.infinity)
            self.lowerBounds.append(0.0)
            self.colName.append("zeta["+str(i+1)+"]")
            self.zetacolName.append("zeta["+str(i+1)+"]")

        for i in range(self.n):
            self.objectiveCoeff.append(1.0)
            self.upperBounds.append(cpx.infinity)
            self.lowerBounds.append(0.0)
            self.colName.append("wPlus["+str(i+1)+"]")
            self.wpluscolName.append("wPlus["+str(i+1)+"]")

        for i in range(self.n):
            self.objectiveCoeff.append(1.0)
            self.upperBounds.append(cpx.infinity)
            self.lowerBounds.append(0.0)
            self.colName.append("wMinus["+str(i+1)+"]")
            self.wminuscolName.append("wMinus["+str(i+1)+"]")

        self.objectiveCoeff.append(0.0)
        self.upperBounds.append(cpx.infinity)
        self.lowerBounds.append(-cpx.infinity)
        self.colName.append("gamma")
        self.gammacolName.append("gamma")
        
        self.p = cpx.Cplex()
        self.p.objective.set_sense(self.p.objective.sense.minimize)
        self.p.variables.add(obj=self.objectiveCoeff,lb=self.lowerBounds,ub=self.upperBounds,names=self.colName)
        
        
    def changeObjectiveCoeffs(self,grad):
        for i in range(self.n):
            self.p.objective.set_linear("w["+str(i+1)+"]", -grad[i])
        
    
    def constraintsMaker(self, ab, lab):
        self.rowName1           = []
        self.rowName2           = []
        
        self.rhs1               = []
        self.rhs2               = []

        self.ranges1            = []
        self.ranges2            = []
        
        self.sense1             = []
        self.sense2             = []
        
        
        for i in range(self.lSize):
            self.rhs1.append(-1.0)
            self.sense1.append("L")
            self.ranges1.append(0.0)
            self.rowName1.append("r1["+str(i+1)+"]")
            
        for i in range(self.mSize):
            self.rhs2.append(-1.0)
            self.sense2.append("L")
            self.ranges2.append(0.0)
            self.rowName2.append("r2["+str(i+1)+"]")
        
        self.matrixMaker(ab, lab)
        
        rows1 = []
        t1 = [0.0 for i in range(len(self.colName))]
        for i in range(self.lSize):
            t1 = self.aMatrix[i].tolist()
            rows1.append([self.colName, t1])
        
        self.p.linear_constraints.add(lin_expr=rows1, rhs=self.rhs1, senses=self.sense1, range_values=self.ranges1, names=self.rowName1)
        
        rows2 = []
        t2 = [0.0 for i in range(len(self.colName))]
        for i in range(self.mSize):
            t2 = self.bMatrix[i].tolist()
            rows2.append([self.colName, t2])
        
        self.p.linear_constraints.add(lin_expr=rows2, rhs=self.rhs2, senses=self.sense2, range_values=self.ranges2, names=self.rowName2)
  
    
    def matrixMaker(self, ab, lab):
                
        self.aMatrix = []
        self.bMatrix = []
        
        ATemp = []
        BTemp = []
        for i in range(0,self.lSize+self.mSize):
            if lab[i] == 0:
                ATemp.append(ab[i])
            else:
                BTemp.append(ab[i])
        
        self.aMatrix = np.array(np.negative(np.identity(self.lSize)))
        self.aMatrix = np.hstack((self.aMatrix,np.zeros((self.lSize,self.mSize))))
        self.aMatrix = np.hstack((self.aMatrix,ATemp))
        self.aMatrix = np.hstack((self.aMatrix,np.negative(ATemp)))
        self.aMatrix = np.hstack((self.aMatrix,np.negative(np.ones((self.lSize,1)))))

        self.bMatrix = np.array(np.zeros((self.mSize,self.lSize)))
        self.bMatrix = np.hstack((self.bMatrix,np.negative(np.identity(self.mSize))))
        self.bMatrix = np.hstack((self.bMatrix,np.negative(BTemp)))
        self.bMatrix = np.hstack((self.bMatrix,BTemp))
        self.bMatrix = np.hstack((self.bMatrix,np.ones((self.mSize,1))))