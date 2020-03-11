import os
import random
import hickle as h
import pickle as p
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from tensorflow.keras import layers
#import keras.backend as K
from tensorflow.keras.models import model_from_json
from Utils.datasets import read_csv
from Utils.features import create_features, padding


def load_data(data):
    '''
    Loads the filenames from data and converts them to features
    Features used are MFCCs.
    '''
    path = 'data/features/'
    mfcc_path = path + 'MFCC/'
    delta_path = path + 'DMFCC/'
    ddelta_path = path + 'DDMFCC/'
    INPUT1 = np.array([])
    INPUT2 = np.array([])
    for i, r in data.iterrows():
        inp1 = r['inp1'].split('.')[0]
        inp2 = r['inp2'].split('.')[0]
        mfcc1 = np.load(mfcc_path + inp1 + '.npy')
        delta1 = np.load(delta_path + inp1 + '.npy')
        ddelta1 = np.load(ddelta_path + inp1 + '.npy')
        mfcc2 = np.load(mfcc_path + inp1 + '.npy')
        delta2 = np.load(delta_path + inp1 + '.npy')
        ddelta2 = np.load(ddelta_path + inp1 + '.npy')
        np.append(INPUT1, np.array([mfcc1, delta1, ddelta1]))
        np.append(INPUT2, np.array([mfcc2, delta2, ddelta2]))
    return (INPUT1, INPUT2)

def Encoder(input_shape, embedding_dimension, drop_out=0.05):
    #Input placeholders
    inp_placeholder = tf.keras.Input(shape=input_shape)
    
    #Convolution layer 1
    layer1 = layers.Conv2D(40, 25, activation='relu', name="L11")(inp_placeholder)
    norm1 = layers.BatchNormalization()(layer1)
    drops1 = layers.SpatialDropout2D(drop_out)(norm1)
    out1 = layers.MaxPooling2D(5)(drops1)
    
    #Convolution layer 2
    layer2 = layers.Conv2D(20, 15, activation='relu', name="L21")(out1)
    norm2 = layers.BatchNormalization()(layer2)
    drops2 = layers.SpatialDropout2D(drop_out)(norm2)
    out2 = layers.MaxPooling2D(2)(drops2)
    
    #Convolution layer 3
    layer3 = layers.Conv2D(10, 5, activation='relu', name="L31")(out2)
    norm3 = layers.BatchNormalization()(layer3)
    out3 = layers.SpatialDropout2D(drop_out)(norm3)
    
    #Flattened layer
    flatten = layers.Flatten()(out3)
    
    #Dense Layer
    embeds = layers.Dense(embedding_dimension, activation = "sigmoid", name="D11")(flatten)
    
    #Definining our model
    encoder = tf.keras.Model(inputs=inp_placeholder, outputs = embeds)
    return encoder

def SiameseNetwork(input_shape):
    inp_1 = tf.keras.Input(shape=input_shape)
    inp_2 = tf.keras.Input(shape=input_shape)
    
    encoder1 = Encoder(input_shape = (5388, 20, 3), embedding_dimension = 128)
    encoder2 = Encoder(input_shape = (5388, 20, 3), embedding_dimension = 128)

    #Encode each branch
    embeds1 = encoder1(inp_1)
    embeds2 = encoder2(inp_2)

    #Siamese network
    embedded_distance = layers.Subtract(name='subtract_embeddings')([embeds1, embeds2])
    embedded_distance1 = layers.Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)), name='euclidean_distance')(embedded_distance)
    siamese_out = layers.Dense(2, activation='sigmoid', name="OutputLayer")(embedded_distance1)

    #Model
    siamesemodel = tf.keras.Model(inputs=[inp_1,inp_2], outputs = siamese_out)
    return (siamesemodel, encoder1, encoder2)


def train_network(model, encoder1, encoder2, train, ytrain, val, yval, signal_data, batchsize, num_epochs, lr=0.001):
    X1 = tf.keras.Input(shape=(5388, 20, 3))
    X2 = tf.keras.Input(shape=(5388, 20, 3))
    y = tf.placeholder('int32',[None],name='t1')
    
    siamese_out = model([X1, X2])

    #Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,logits = siamese_out)
    loss = tf.reduce_mean(loss)
    
    #Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    soft_out = tf.nn.softmax(siamese_out)
    

    y_onehot = tf.one_hot(y, 2)
    
    #Prediction and accuracy
    pred = tf.equal(tf.argmax(soft_out,1), tf.argmax(y_onehot,1))
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
    
    #Saving history
    hist = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #steps = train.shape[0]//batchsize
        steps = 1000
        print("Steps:", steps)
        for epoch in range(num_epochs):
            epoch_loss1 = 0
            epoch_acc1 = 0
            epoch_loss2 = 0
            epoch_acc2 = 0
            for step in range(steps):
                start = step*batchsize
                end = min(start + batchsize, train.shape[0])
                batch_x = train[start:end]
                #batch_data1 = np.array([signal_data[i[0]] for i in batch_x])
                #batch_data2 = np.array([signal_data[i[1]] for i in batch_x])
                batch_data1, batch_data2 = load_data(batch_x)
                batchy = ytrain[start:end]
                #print(batch_data1.shape, batch_data2.shape, batchy.shape)
                _, cost = sess.run([optimizer, loss], feed_dict={X1:batch_data1, X2:batch_data2, y:batchy})
                train_acc = sess.run(accuracy, feed_dict = {X1:batch_data1, X2:batch_data2, y:batchy})
                epoch_loss1 += cost
                epoch_acc1 += train_acc
                #if step%100 == 0:
                # print("Epoch:",epoch,"Step:",step+1,"TrainLoss:",epoch_loss1/(step+1), "accuracy:", epoch_acc1/(step+1))
                
                
                #Reverse
                batch_data1 = np.array([signal_data[i[1]] for i in batch_x])
                batch_data2 = np.array([signal_data[i[0]] for i in batch_x])
                batchy = ytrain[start:end]
                #print(batch_data1.shape, batch_data2.shape, batchy.shape)
                _, cost = sess.run([optimizer, loss], feed_dict={X1:batch_data1, X2:batch_data2, y:batchy})
                train_acc = sess.run(accuracy, feed_dict = {X1:batch_data1, X2:batch_data2, y:batchy})
                epoch_loss2 += cost
                epoch_acc2 += train_acc
                if step%100 == 0:
                    print("Epoch:",epoch,"Step:",step+1,"TrainLoss:",(epoch_loss1/(step+1) + epoch_loss2/(step+1))/2, "accuracy:", (epoch_acc2/(step+1) + epoch_acc2/(step+1))/2)
                hist['train_loss'].append((epoch_loss1/(step+1) + epoch_loss2/(step+1))/2)
                hist['train_acc'].append((epoch_acc1/(step+1) + epoch_acc2/(step+1))/2)
                
        spkModel_json = model.to_json()
        ebdModel1_json = encoder1.to_json()
        ebdModel2_json = encoder2.to_json()
        with open("Smodel" + str(epoch) + ".json", "w") as json_file:
        json_file.write(spkModel_json)
        
        with open("Emodel" + str(epoch) + ".json", "w") as json_file:
        json_file.write(ebdModel1_json)
        
        with open("Emodel" + str(epoch) + ".json", "w") as json_file:
        json_file.write(ebdModel2_json)
        
        model.save_weights("Smodel"+str(epoch)+".h5")
        encoder1.save_weights("Emodel"+str(epoch)+".h5")
        encoder2.save_weights("Emodel"+str(epoch)+".h5")
          
        print("Saved model to disk")
        #print("Epoch:",epoch,"Loss:", epoch_loss/steps, "accuracy:", epoch_acc/steps)
        '''
        val_batches = val.shape[0]//batchsize
        val_l = 0
        val_a = 0
        for x in range(val_batches):
            start = x*batchsize
            end = min(start + batchsize, val.shape[0])
            valbatch_x = val[start:end]
            valbatch_data1 = np.array([signal_data[i[0]] for i in valbatch_x])
            valbatch_data2 = np.array([signal_data[i[1]] for i in valbatch_x]) 
            valbatchy = yval[start:end]
            val_loss, val_acc = sess.run([loss,accuracy], feed_dict = {X1:valbatch_data1,
                                                                        X2:valbatch_data2,
                                                                        y:valbatchy})
            val_l += val_loss
            val_a += val_acc
            hist['val_loss'].append(val_loss)
            hist['val_acc'].append(val_acc)
        print("Validation:","Epoch:",epoch,"Loss:", val_l/val_batches, "accuracy:", val_a/val_batches)
        '''  
    return (hist, model, encoder1, encoder2)

if __name__ == "__main__":
    train = sys.argv[1]
    val = sys.argv[2]
    
