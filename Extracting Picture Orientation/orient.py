#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import time as t
import pickle
import math
import pandas as pd
#%%

def read_traindata(train):
    #extracting train data
    train_file = open(train,'r',encoding = 'utf-8')
    train_data = [i.split() for i in train_file.read().split('\n') if len(i) > 0]
    Xtrain = [x[2:] for x in train_data]
    Xtrain = [[int(y) for y in x[2:]] for x in train_data]
    ytrain = [x[1] for x in train_data]
    training_data = [(i,j) for i,j in zip(Xtrain,ytrain)]
    return Xtrain,ytrain,training_data
    
def read_testdata(test):
    #extracting test data
    test_file = open(test,'r',encoding = 'utf-8')
    test_data = [i.split() for i in test_file.read().split('\n') if len(i) > 0]
    Xtest = [x[2:] for x in test_data]
    Xtest = [[int(y) for y in x[2:]] for x in test_data]
    ytest = [x[1] for x in test_data]
    test_label=[x[0] for x in test_data]
    return Xtest,ytest,test_label

class Tree(object):
    def __init__(self):
        self.data = None
        self.left = None
        self.right = None

class Random_Forest():
    
    def calc_entropy(self,data):
        if len(data) == 0:
            return 0
        pixel_dict={'0':0,'90':0,'180':0,'270':0}
        for i in data:
            pixel_dict[i[1]] += 1
        ent = 0
        for key in pixel_dict:
            prob = pixel_dict[key]/sum(pixel_dict.values())
            if prob == 0:
                continue
            val = -prob * np.log(prob)
            ent += val
        return ent
    
    def choose_node(self,data,root,num_split,entropy,pixel_list,depth,partition):
        pixels_left = [i for i in range(0,192) if i not in pixel_list]
        if num_split > depth or entropy==0 or len(pixels_left)==0 :
            data_dict = {}
            for i in data:
                if i[0][1] not in data_dict.keys():
                    data_dict[i[1]] = 1
                else:
                    data_dict[i[1]] += 1
            max_value = 0
            max_key = 0
            for key in data_dict.keys():
                if data_dict[key] > max_value:
                    max_value = data_dict[key]
                    max_key = key
            root.data = ("Angle:"+str(max_key),partition)        
            return
        
        if len(data)==0:
            return 

        entropy_final = sys.maxsize
        pixel_chosen = -1
        for p in pixels_left:
            for partition in [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250]:
                #splitting on this pixel
                left_data = [i for i in data if i[0][p] < partition]
                right_data = [i for i in data if i[0][p] >= partition]
                #calculating entropy for this split
                entropy = (len(left_data)/len(data)*self.calc_entropy(left_data))\
                          + (len(right_data)/len(data)*self.calc_entropy(right_data))
                #storing the best split
                if entropy_final > entropy:
                    entropy_final = entropy
                    pixel_chosen = p
                    left= left_data
                    right=right_data
                    partition_chosen = partition
        #adding the pixel that was chosen to split above, so that  this pixel is not chosen again
        pixel_list.append(p)
        #adding left and right trees
        root.left = Tree()
        root.right = Tree()
        root.data = (pixel_chosen,partition_chosen)
        #incrementing the level of tree
        num_split += 1
        #recursive call for left and right half
        self.choose_node(left,root.left,num_split,entropy_final,pixel_list,depth,partition_chosen) 
        self.choose_node(right,root.right,num_split,entropy_final,pixel_list,depth,partition_chosen)
        #return the root of the tree
        return root
    
    def train(self,data,num_trees,depth):
        root_list = []
        x = num_trees
        while(num_trees>0):
            print("Tree:",x-num_trees)
            root = Tree()
            pixel_list = []
            #taking a random sample set from training set
            random_samples = list(np.random.choice(np.arange(len(data)),2*len(data)//3))
            samples = [data[i] for i in random_samples]
            root = self.choose_node(samples,root,0,1,pixel_list,depth,-1)
            root_list.append(root)
            num_trees -= 1
        return root_list

    def make_pred(self,root_list,Xtest):
        pred = []
        for i in Xtest:
            label = []
            for root in root_list:
                node = root
                while(1):
                    if node.data == None:
                        label.append('NA')
                        break
                    elif len(str(node.data)) > 10:
                        label.append(node.data[0][6:])
                        break
                    else:    
                        split_info = node.data
                        if int(i[split_info[0]]) < split_info[1]:
                            node = node.left
                        else:
                            node = node.right
            pred.append(label)
        pred_dict = {}
        for i in range(len(pred)):
            if i not in pred_dict.keys():
                pred_dict[i] = {}
            for j in pred[i]:
                if j not in pred_dict[i].keys():
                    pred_dict[i][j] = 1
                else:
                    pred_dict[i][j] += 1
                
        prediction = []
        for k in pred_dict.keys():
            prediction.append(max(pred_dict[k],key=lambda x: pred_dict[k][x]))
        return prediction                


#%%
class Adaboost():
    def weight_update(self,w,indices,error):
        for i in range (len(w)):
            w[i] = w[i]*error/(1.0000000000002651-error)
        sum_w = sum(w)
        w = [i/sum_w for i in w]
        return w
    
    def make_stump(self,pixel):
        root = Tree()
        root.data = (pixel,pixel+4)
        root.left = 1
        root.right = -1
        return root
    
    def most_common_label(self,data):
        label_dict = {'0':0,'90':0,'180':0,'270':0}
        for i in data:
            label_dict[i[1]] += 1
        return max(label_dict,key=lambda x: label_dict[x])
        
    def train(self,data,num_trees):
        l = len(data)
        w = [1/l for i in range(len(data))]
        print(sum(w))
        pixel_list = [i for i in range(0,192)]
        it = 0
        root_list = []
        alpha_list = []
        x = num_trees
        while(x>0):    
            pixel = np.random.choice(pixel_list[:-4],1)[0]
            error = 0
            min_error = sys.maxsize
            for ornt in ['0','90','180','270']:
                mislabeled_indices = []
                for i in range(len(data)):
                    if data[i][0][pixel]-data[i][0][pixel+4] < 40:
                        if data[i][1] != ornt:
                            mislabeled_indices.append(i)
                    else:
                        if data[i][1] == ornt:
                            mislabeled_indices.append(i)
                #print('Indices:',len(mislabeled_indices))
                for i in mislabeled_indices:
                    error += w[i]
                #print(error)
                if error > 1:
                    error = 1
                w = self.weight_update(w,mislabeled_indices,min_error)
                root_list.append(self.make_stump(pixel))
                alpha_list.append(math.log((1.0000000000002651-error)/error))
            print("Stump",it,"done...")  
            it += 1
            x-=1
        return root_list,alpha_list    
        
    def make_pred(self,data,all_roots,all_alpha):
        prediction = []
        votes = []
        roots_0 = [i for i in all_roots if all_roots.index(i)%4==0]
        roots_90 = [i for i in all_roots if all_roots.index(i)%4==1]
        roots_180 = [i for i in all_roots if all_roots.index(i)%4==2]
        roots_270 = [i for i in all_roots if all_roots.index(i)%4==3]
        
        alpha_0 = [i for i in range(len(all_alpha)) if i%4==0]
        alpha_90 = [i for i in range(len(all_alpha)) if i%4==1]
        alpha_180 = [i for i in range(len(all_alpha)) if i%4==2]
        alpha_270 = [i for i in range(len(all_alpha) )if i%4==3]
    
        for img in data:
            pred_votes = []
            score = 0
            votes = [0,0,0,0]
            for i in range(len(all_roots)):
                # print(i)
                 if i%4 == 0:
                    pix1,pix2 = roots_0[i//4].data[0],roots_0[i//4].data[1]
                    diff = pix1-pix2
                    if diff < 40:
                        votes[0] += alpha_0[i//4]*roots_0[i//4].left
                    else:
                        votes[0] += alpha_0[i//4]*roots_0[i//4].right
                 if i%4 == 1:
                    pix1,pix2 = roots_90[i//4].data[0],roots_90[i//4].data[1]
                    diff = pix1-pix2
                    if diff < 40:
                        votes[1] += alpha_90[i//4]*roots_90[i//4].left
                    else:
                        votes[1] += alpha_90[i//4]*roots_90[i//4].right
                 if i%4 == 2:
                    pix1,pix2 = roots_180[i//4].data[0],roots_180[i//4].data[1]
                    diff = pix1-pix2
                    if diff < 40:
                        votes[2] += alpha_180[i//4]*roots_180[i//4].left
                    else:
                        votes[2] += alpha_180[i//4]*roots_180[i//4].right
                 if i%4 == 3:
                    pix1,pix2 = all_roots[i//4].data[0],all_roots[i//4].data[1]
                    diff = pix1-pix2
                    if diff < 40:
                        votes[3] += alpha_270[i//4]*roots_270[i//4].left
                    else:
                        votes[3] += alpha_270[i//4]*roots_270[i//4].right
            if votes.index(max(votes)) == 0:
                prediction.append('0')
            elif votes.index(max(votes)) == 1:
                prediction.append('90')
            elif votes.index(max(votes)) == 2:
                prediction.append('180')
            else:
                prediction.append('270')
            pred_votes.append(votes)
        return prediction,votes

#%%
class KNN():
    #This function calculates the euclidean distance between two points
    def eu_dis(self,point1, point2):
        distance=np.linalg.norm(np.array(point1)-np.array(point2))
        return distance

    
    #This function calculates the nearest k neighbor and predict the orientation of the testdata
    #by max votes approach.
    #for example if the value of K is 5 and the algorithm computes the 5 nearest training points 
    #and then checks the orientation of these 5 points, consider the following is the orientation
    #of 5 data points (180,180,180,0,0), then predicted orientation of testpoint would be 180
    
    def make_pred(self,test,X,y,k):
        distances =[]
        #Calculating the euclidean distance, for given test data with all the traindataset.
        for i in range(len(Xtrain)):
            dist = self.eu_dis(test, X[i])
            distances.append((y[i], dist))
        orient_near=[]
        #this loop, find the minimun distance point and then pop out this value and then computes
        #the next minimum value and this process continues for K times
        for i in range(k):
            min_value= min(distances,key=lambda item:item[1])
            distances=[i for i in distances if i[1] > min_value[1]]
            orient_near.append(min_value[0])
        #Picking up the orientation based on max count.    
        Pred = max(orient_near, key=orient_near.count)       
        return Pred

#%%
mode = sys.argv[1]
inp_filename = sys.argv[2]
model_filename = sys.argv[3]
model = sys.argv[4]
if mode == 'train':
    Xtrain,ytrain,training_data = read_traindata(inp_filename)
    if model == 'nearest' or model == 'best':
        K = KNN()
        file = open(model_filename,"w+",encoding = 'utf-8')
        file.write('11')
        file.close
    if model == 'forest':
        R = Random_Forest() 
        s = t.time()
        root_list = R.train(training_data,100,15)
        e = t.time()
        print('Training Time:',(e-s)/60,"min")
        file = open(model_filename,"wb+")
        pickle.dump(root_list,file)
        file.close()
    if model == 'adaboost': 
        Ada = Adaboost()
        s = t.time()
        root,alpha = Ada.train(training_data,200)
        e = t.time()
        print("Training time:",(e-s)/60,"min")
        file = open(model_filename,"wb+")
        data = [(i,j) for i,j in zip(root,alpha)]
        pickle.dump(data,file)
        file.close()

if mode == 'test':
    Xtest,ytest,test_label = read_testdata(inp_filename)
    if model == 'nearest' or model == 'best':
        K=KNN()
        file = open(model_filename,"r",encoding ="utf-8")
        x = int(file.read())
        file.close()
        s = t.time()
        y_pred=[]
        count=0
        for i in range(len(Xtest)):
            pred=K.make_pred(Xtest[i],Xtrain,ytrain,x)
            y_pred.append(pred)
        e = t.time()
        file = open('Output.txt','w+',encoding = 'utf-8')
        for i in range(len(y_pred)):
            if y_pred[i] == ytest[i]:
                count += 1
            file.write(test_label[i] + " " + y_pred[i] + "\n")
        file.close()
        print("Testing time:",e-s,"sec")     
        print("Accuracy:",count/len(y_pred)*100,"%")
    if model == 'forest':
        R = Random_Forest()
        file=open(model_filename,"rb")
        root = pickle.load(file)
        file.close()
        s = t.time()    
        y_pred = R.make_pred(root,Xtest)
        e = t.time()
        output = []
        count = 1
        #Opening file to write
        file = open("Output.txt","w+",encoding = "utf-8")
        for i in range(len(y_pred)):
            if y_pred[i] == ytest[i]:
                count += 1
            file.write(test_label[i] + " " + str(y_pred[i]) + "\n")
        file.close()
        print("Testing time:",e-s,"sec")
        print("Accuracy:",count/len(y_pred)*100,"%")
    if model == 'adaboost':
        Ada = Adaboost()
        file=open(model_filename,"rb")
        val = pickle.load(file)
        root = [i[0] for i in val]
        alpha = [i[1] for i in val]
        file.close()
        s = t.time()    
        y_pred,votes = Ada.make_pred(Xtest,root,alpha)
        e = t.time()
        output = []
        count = 1
        #Opening file to write
        file = open("Output.txt","w+",encoding = "utf-8")
        for i in range(len(y_pred)):
            if y_pred[i] == ytest[i]:
                count += 1
            file.write(test_label[i] + " " + str(y_pred[i]) + "\n")
        file.close()
        print("Testing time:",e-s,"sec")
        print("Accuracy:",count/len(y_pred)*100,"%")