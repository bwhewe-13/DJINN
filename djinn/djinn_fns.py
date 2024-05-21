###############################################################################
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
#
# Written by K. Humbird (humbird1@llnl.gov), L. Peterson (peterson76@llnl.gov).
#
# LLNL-CODE-754815
#
# All rights reserved.
#
# This file is part of DJINN.
#
# For details, see github.com/LLNL/djinn. 
#
# For details about use and distribution, please read DJINN/LICENSE .
###############################################################################

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import sys
import numpy as np
from sklearn.tree import _tree
from sklearn.preprocessing import MinMaxScaler
try: from sklearn.model_selection import train_test_split
except: from sklearn.cross_validation import train_test_split
try: import cPickle
except: import _pickle as cPickle
    
        
def tree_to_nn_weights(regression, X, Y, num_trees, rfr, random_state) :
    """ Main function to map tree to neural network. Determines architecture, initial weights.  
        
    Args:
        x (ndarray): Input features.
        y (ndarray): Output features.
        ntrees (int): Number of decision trees.
        rfr (object): Random forest regressor.
        random_state (int): Sets random seed.

    Returns:
        dict: includes weights, biases, architecture
    """

    def xav(nin,nout):
        """Xavier initialization. args: input & output dim of layer """
        return(np.random.normal(loc=0.0,scale=np.sqrt(3.0/(nin+nout))))

    #set seed
    if random_state: np.random.seed(random_state)

    #get input & output dimensions from data
    nin = X.shape[1]
    if regression == True:
        if(Y.size > Y.shape[0]): nout = Y.shape[1]
        else: nout = 1

    else: 
        nout=len(np.unique(Y))

    #store nn info to pass to tf
    tree_to_network={}
    tree_to_network['n_in'] = nin
    tree_to_network['n_out'] = nout
    tree_to_network['network_shape']={}
    tree_to_network['weights']={}
    tree_to_network['biases']={}

    #map each tree in RF to initial weights
    for tree in range(num_trees):
        #pull data from tree
        tree_ = rfr.estimators_[tree].tree_
        features=tree_.feature
        n_nodes = tree_.node_count
        children_left = tree_.children_left
        children_right = tree_.children_right
        threshold = tree_.threshold
        
        # Traverse tree structure to record node depth, node id, etc
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
        
        
        # collect tree structure in dict, sorted by node number
        node={}
        for i in range(len(features)):
            node[i]={}
            node[i]['depth']=node_depth[i]
            if(features[i]>=0): node[i]['feature']=features[i]
            else: node[i]['feature']=-2
            node[i]['child_left']=features[children_left[i]]
            node[i]['child_right']=features[children_right[i]]
            
            
        # meta data arrays for mapping to djinn weights
        num_layers=len(np.unique(node_depth))  #number of layers in nn  
        nodes_per_depth=np.zeros(num_layers)   #number nodes in each layer of tree
        leaves_per_depth=np.zeros(num_layers)  #number leaves in each layer of tree  
        
        for i in range(num_layers):
            ind=np.where(node_depth==i)[0]
            nodes_per_depth[i]=len(np.where(features[ind]>=0)[0])
            leaves_per_depth[i]=len(np.where(features[ind]<0)[0])    
    
        #max depth at which each feature appears in tree
        max_depth_feature=np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            ind=np.where(features==i)[0]
            if(len(ind)>0): max_depth_feature[i]=np.max(node_depth[ind])
        
        #djinn architecture    
        djinn_arch=np.zeros(num_layers, dtype='int')  
    
        #hidden layers = previous layer + # new nodes in layer of tree 
        djinn_arch[0]=nin
        for i in range(1,num_layers):
            djinn_arch[i]=djinn_arch[i-1]+nodes_per_depth[i]
        djinn_arch[-1]=nout
    
        #create dict for djinn weights : create blank arrays
        djinn_weights={}
        for i in range(num_layers-1):
            djinn_weights[i]=np.zeros((djinn_arch[i+1],djinn_arch[i]))
        
        #create list of indices for new neurons in each layer    
        new_n_ind=[]
        for i in range(num_layers-1):
            new_n_ind.append(np.arange(djinn_arch[i],djinn_arch[i+1]))  
            
        #fill in weights in djinn arrays
        for i in range(num_layers-1): #loop through layers
            nn_in = djinn_weights[i].shape[0]
            nn_out = djinn_weights[i].shape[1]
            for f in range(nin): 
                #add diagonal terms up to depth feature is used
                if(i < max_depth_feature[f]-1): djinn_weights[i][f,f]=1.0
            #begin mapping off diagonal connections
            k=0; kk=0; #k keeps track of outgoing layer, kk keeps track of incoming layer neuron index
            for nodes in node:
                if node[nodes]['depth']== i :
                    feature=node[nodes]['feature']
                    if feature >= 0: #if node is a split/not a leaf
                        left=node[nodes]['child_left']
                        right=node[nodes]['child_right']
                        if((nodes==0) and ((left<0) or (right<0)) ): 
                            #leaf at first split: connect through out layer
                            for j in range(i,num_layers-2): djinn_weights[j][feature,feature]=1.0 
                            djinn_weights[num_layers-2][:,feature]=1.0
                        if(left>=0): 
                            #left child is split, connect nodes in that decision 
                            #to new neuron in current nn layer
                            if(i==0): djinn_weights[i][new_n_ind[i][k],feature] = xav(nn_in,nn_out)
                            else: djinn_weights[i][new_n_ind[i][k],new_n_ind[i-1][kk]] = xav(nn_in,nn_out)
                            djinn_weights[i][new_n_ind[i][k],left] = xav(nn_in,nn_out)
                            k+=1
                            if( kk >= len(new_n_ind[i-1]) ): kk=0 
                        if( (left<0) and (nodes!=0) ): #leaf- connect through to outputs
                            lind=new_n_ind[i-1][kk]
                            for j in range(i,num_layers-2): djinn_weights[j][lind,lind]=1.0 
                            djinn_weights[num_layers-2][:,lind]=1.0
                        if(right>=0):
                            #right child is split, connect nodes in that decision 
                            #to new neuron in current nn layer
                            if(i==0): djinn_weights[i][new_n_ind[i][k],feature]=xav(nn_in,nn_out)
                            else: djinn_weights[i][new_n_ind[i][k],new_n_ind[i-1][kk]]=xav(nn_in,nn_out)
                            djinn_weights[i][new_n_ind[i][k],right]=xav(nn_in,nn_out)
                            k+=1
                            if( kk >= len(new_n_ind[i-1])): kk=0 
                        if( (right<0) and (nodes!=0) ): #leaf- connect through to outputs
                            lind=new_n_ind[i-1][kk]
                            for j in range(i,num_layers-2): djinn_weights[j][lind,lind]=1.0 
                            djinn_weights[num_layers-2][:,lind]=1.0
                        kk+=1    
    
        #connect active neurons to output layer                         
        m=len(new_n_ind[-2])
        ind=np.where(abs(djinn_weights[num_layers-3][:,-m:])>0)[0]
        for inds in range(len(djinn_weights[num_layers-2][:,ind])):
            djinn_weights[num_layers-2][inds,ind]=xav(nn_in,nn_out)
    
        # dump weights, arch, biases into dict to pass to tf
        tree_to_network['network_shape']['tree_%s'%tree] = djinn_arch
        tree_to_network['weights']['tree_%s'%tree] = djinn_weights
        tree_to_network['biases']['tree_%s'%tree] = [] #maybe add biases

    return tree_to_network
 
    


def tf_dropout_regression(regression, ttn, xscale, yscale, x1, y1, ntrees, filename, 
                          learnrate, training_epochs, batch_size,
                          dropout_keep_prob, weight_reg, display_step, savefiles, 
                          savemodel, modelname, modelpath, random_state):
    """ Trains neural networks in tensorflow, given initial weights from decision tree.
        
    Args:
        ttn (dict): Dictionary returned from function tree_to_nn_weights.
        xscale (object): Input scaler.
        yscale (object): Output scaler.
        x1 (ndarray): Input features.
        y1 (ndarray): Output features.
        ntrees (int): Number of decision trees.
        filename (str): Name for saved files.
        learn_rate (float): Learning rate for optimizaiton of weights/biases.
        training_epochs (int): Number of epochs to train neural network.
        batch_size (int): Number of samples per batch.
        dropout_keep_prob (float): Probability of keeping neuron "on" in dropout layers.
        weight_reg (float): Multiplier for L2 penalty on weights.
        display_step (int): Cost is printed every display_steps during training.
        save_files (bool): If True, saves train/valid cost per epoch, weights/biases.
        file_name (str): File name used if 'save_files' is True.
        save_model (bool): If True, saves the trained model.
        model_name (str): File name for model if 'save_model' is True.
        model_path (str): Location of where the model/files are saved. 
        random_state (int): Sets random seed.

    Returns:
        dict: final neural network info: weights, biases, cost per epoch.
    """
    #get size of input/output layer
    n_input = ttn['n_in']    
    n_classes = ttn['n_out']
    #save min/max values for python-only djinn eval
    input_min = np.min(x1, axis=0)
    input_max = np.max(x1, axis=0)
    if(n_classes == 1): y1 = y1.reshape(-1,1)
    output_min = np.min(y1, axis=0)
    output_max = np.max(y1, axis=0)

    #scale data
    x1 = xscale.transform(x1)
    if regression == True:
        if(n_classes == 1): y1 = yscale.transform(y1).flatten()
        else: y1 = yscale.transform(y1)

    xtrain, xtest, ytrain, ytest = train_test_split(x1, y1, test_size=0.1, random_state=random_state) 

    #for classification, do one-hot encoding on classes
    if regression == False:
        ytrain=np.equal.outer(ytrain.flatten(), np.arange(len(np.unique(ytrain)))).astype("float32")
        ytest=np.equal.outer(ytest.flatten(), np.arange(len(np.unique(ytest)))).astype("float32")

    pp = 0
    #create dict/arrays to save network info
    nninfo = {}
    nninfo['input_minmax'] = [input_min, input_max]
    nninfo['output_minmax'] = [output_min, output_max]
    nninfo['initial_weights'] = {}; nninfo['initial_biases'] = {}; 
    nninfo['final_weights'] = {}; nninfo['final_biases'] = {}; 
    train_accur = np.zeros((ntrees, training_epochs))
    valid_accur = np.zeros((ntrees, training_epochs))
    #loop through trees, training each network in ensemble
    for keys in ttn["weights"]:
        tf.keras.backend.clear_session()
        if (random_state): tf.random.set_seed(random_state) 


        #get network shape from djinn mapping
        npl = ttn['network_shape'][keys]
        nhl = len(npl)-2
        n_hidden = {}
        for i in range(1, len(npl)-1):
            n_hidden[i] = npl[i]

        
        def multilayer_perceptron(weights, biases):
            layer = []
            regularizer = tf.keras.regularizers.L2(l2=.5*weight_reg)
            w = WB_Init(weights['h1'],name='w1')
            b = WB_Init(biases['h1'],name='b1')
            layer.append(tf.keras.layers.Dense(weights['h1'].shape[1],
                                               activation='relu',
                                               kernel_initializer = w,
                                               bias_initializer = b,
                                               kernel_regularizer = regularizer))
            for i in range(2,nhl+1):
                w = WB_Init(weights['h%s'%i],name='w%s'%i)
                b = WB_Init(biases['h%s'%i],name = 'b%s'%i)
                layer.append(tf.keras.layers.Dense(weights['h%s'%i].shape[1],
                                                       activation='relu', kernel_initializer=w,
                                                       bias_initializer = b,
                                                       kernel_regularizer = regularizer))
                layer.append(tf.keras.layers.Dropout(1.0-dropout_keep_prob,trainable=True))
            w = WB_Init(weights['out'])
            b = WB_Init(biases['out'])
            out_layer = tf.keras.layers.Dense(weights['out'].shape[1],
                                              kernel_initializer = w,
                                              bias_initializer=b,
                                              kernel_regularizer=regularizer,
                                              name='prediction')
            inputs = tf.keras.Input((n_input,))
            x_in = inputs
            for i in layer:
                x_in = i(x_in)
            out_final = out_layer(x_in)
            model = tf.keras.Model(inputs,out_final,name="DJINN_MODEL")
            return model

            #get weights from djinn mapping; biases are random
        w = {}; b = {};
        for i in range(0, len(ttn['network_shape'][keys])-1):
            w[i+1] = np.transpose(ttn['weights'][keys][i]).astype((np.float32))

        weights = {}
        for i in range(1, nhl+1):
            weights['h%s'%i] = w[i]
            
        weights['out'] = w[nhl+1]
            
            
        biases = {}
        for i in range(1, nhl+1):
            biases['h%s'%i] = np.random.normal(loc=0.0,scale=np.sqrt(3.0/(n_classes+int(n_hidden[nhl]))),size=(int(n_hidden[i]),))
        biases['out'] = np.random.normal(loc=0.0,scale=np.sqrt(3.0/(n_classes+int(n_hidden[nhl]))),size=(n_classes,))  

        pred = multilayer_perceptron(weights, biases)
       
            
        if regression==False:
            cost = tf.keras.losses.CategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,name='cost')
            accuracy = tf.keras.metrics.CategoricalAccuracy(dtype='float32')
            accuracy_label = ['categorical_accuracy']
        else:
            cost = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,name='cost')
            accuracy = None
            accuracy_label = None
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learnrate,name='opt')
            

        pred.compile(optimizer=optimizer,loss=cost,metrics=accuracy_label)
        
        denseWeights = []
        denseBiases = []
        weightsAndBiases = pred.get_weights()
        for i in range(int(len(weightsAndBiases)*.5)):
            denseWeights.append(weightsAndBiases[2*i])
            denseBiases.append(weightsAndBiases[2*i+1])
        
        
        nninfo['initial_weights'][keys] = denseWeights
        nninfo['initial_biases'][keys] = denseBiases
        
        history = pred.fit(xtrain,ytrain,batch_size=batch_size,epochs=training_epochs,verbose=2,validation_data=(xtest,ytest))
        train_accur = history.history['loss']
        valid_accur = history.history['val_loss']
        print("Optimization Finished!")
        
        #save final weights/biases
        denseWeights = []
        denseBiases = []
        weightsAndBiases = pred.get_weights()
        for i in range(int(len(weightsAndBiases)*.5)):
            denseWeights.append(weightsAndBiases[2*i])
            denseBiases.append(weightsAndBiases[2*i+1])

        nninfo['final_weights'][keys] = denseWeights
        nninfo['final_biases'][keys] = denseBiases
        
        #save model
        if(savemodel == True):
            save_path = "%s%s_tree%s.ckpt"%(modelpath, modelname, pp)
            pred.save(save_path,save_format='h5')
            print("Model saved in: %s" % save_path)
        pp += 1 
   
    #save files with nn info  
    nninfo['train_cost'] = train_accur
    nninfo['valid_cost'] = valid_accur
    if(savefiles == True):        
        with open('%snn_info_%s.pkl'%(modelpath, filename), 'wb') as f1:
            cPickle.dump(nninfo, f1)
    return(nninfo)


def get_hyperparams(regression, ttn, xscale, yscale, x1, y1, dropout_keep_prob, 
                    weight_reg, random_state):
    """Performs search for automatic selection of djinn hyper-parameters.
        
    Returns learning rate, number of epochs, batch size.
        
    Args: 
        ttn (dict): Dictionary returned from function tree_to_nn_weights.
        xscale (object): Input scaler.
        yscale (object): Output scaler.
        x1 (ndarray): Input features.
        y1 (ndarray): Output features.
        dropout_keep_prob (float): Probability of keeping neuron "on" in dropout layers.
        weight_reg (float): Multiplier for L2 penalty on weights.
        random_state (int): Sets random seed.

        Returns: 
            dictionary: Dictionary with batch size, 
                        learning rate, number of epochs
        """
    tf.keras.backend.clear_session()
    #get size of input/output
    n_input = ttn['n_in']    
    n_classes = ttn['n_out']

    #scale data
    x1 = xscale.transform(x1)
    if regression == True: 
        if(n_classes == 1): y1 = yscale.transform(y1).flatten()
        else: y1 = yscale.transform(y1)  


    xtrain, xtest, ytrain, ytest = train_test_split(x1, y1, test_size=0.1, random_state=random_state) 

    #for classification, do one-hot encoding on classes
    if regression == False:
        ytrain=np.equal.outer(ytrain.flatten(), np.arange(len(np.unique(ytrain)))).astype("float32")
        ytest=np.equal.outer(ytest.flatten(), np.arange(len(np.unique(ytest)))).astype("float32")

    ystar = {}
    ystar['preds'] = {}

    #learning rates to test
    minlr = -4.0
    maxlr = -2.0
    lrs = np.logspace(minlr, maxlr, 10)

    #default batch size 
    batch_size = int(np.ceil(0.05*len(y1)))

    #optimizing on one tree only
    keys = 'tree_0'
    print("Determining learning rate...")
    for lriter in range(0, 2): #iterate twice
        #arrays for performance data
        accur = np.zeros((len(lrs), 100))
        errormin = np.zeros(len(lrs))
        for pp in range(0, len(lrs)):

            npl = ttn['network_shape'][keys]
            nhl = len(npl)-2
            n_hidden = {}
            for i in range(1, len(npl)-1):
                n_hidden[i] = npl[i]

        
            def multilayer_perceptron(weights, biases):
                layer = []
                regularizer = tf.keras.regularizers.L2(l2=.5*weight_reg)
                w = WB_Init(weights['h1'],name='w1')
                b = WB_Init(biases['h1'],name='b1')
                layer.append(tf.keras.layers.Dense(weights['h1'].shape[1],
                                               activation='relu',
                                               kernel_initializer = w,
                                               bias_initializer = b,
                                               kernel_regularizer = regularizer))
                for i in range(2,nhl+1):
                    w = WB_Init(weights['h%s'%i],name='w%s'%i)
                    b = WB_Init(biases['h%s'%i],name = 'b%s'%i)
                    layer.append(tf.keras.layers.Dense(weights['h%s'%i].shape[1],
                                                       activation='relu', kernel_initializer=w,
                                                       bias_initializer = b,
                                                       kernel_regularizer = regularizer))
                    layer.append(tf.keras.layers.Dropout(1.0-dropout_keep_prob,trainable=True))
                w = WB_Init(weights['out'])
                b = WB_Init(biases['out'])
                out_layer = tf.keras.layers.Dense(weights['out'].shape[1],
                                              kernel_initializer = w,
                                              bias_initializer=b,
                                              kernel_regularizer=regularizer,
                                              name='prediction')
                inputs = tf.keras.Input((n_input,))
                x_in = inputs
                for i in layer:
                    x_in = i(x_in)
                out_final = out_layer(x_in)
                model = tf.keras.Model(inputs,out_final,name="DJINN_MODEL")
                return model

            #get weights from djinn mapping; biases are random
            w = {}; b = {};
            for i in range(0, len(ttn['network_shape'][keys])-1):
                w[i+1] = np.transpose(ttn['weights'][keys][i]).astype((np.float32))

            weights = {}
            for i in range(1, nhl+1):
                weights['h%s'%i] = w[i]
            
            weights['out'] = w[nhl+1]
            
            
            biases = {}
            for i in range(1, nhl+1):
                biases['h%s'%i] = np.random.normal(loc=0.0,scale=np.sqrt(3.0/(n_classes+int(n_hidden[nhl]))),size=(int(n_hidden[i]),))
            biases['out'] = np.random.normal(loc=0.0,scale=np.sqrt(3.0/(n_classes+int(n_hidden[nhl]))),size=(n_classes,))  

            pred = multilayer_perceptron(weights, biases)
            
            
            if regression==False:
                cost = tf.keras.losses.CategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,name='cost')
                accuracy = tf.keras.metrics.CategoricalAccuracy(dtype='float32')
            else:
                cost = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,name='cost')
                accuracy = None
            optimizer = tf.keras.optimizers.Adam(learning_rate=lrs[pp],name='opt')


            pred.compile(optimizer=optimizer,loss=cost,metrics=accuracy)
            history = pred.fit(xtrain,ytrain, batch_size=batch_size, epochs=100,verbose=0)
            accur[pp] = history.history['loss']
            
            errormin[pp] = np.mean(accur[pp, 90:])
            
        indices = errormin.argsort()[:2]
        minlr = np.min((lrs[indices[0]], lrs[indices[1]])) 
        maxlr = np.max((lrs[indices[0]], lrs[indices[1]])) 
        lrs = np.linspace(minlr, maxlr, 10)

    learnrate = minlr 



    print("Determining number of epochs needed...")
    training_epochs = 3000; pp = 0;
    accur = np.zeros((1, training_epochs))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learnrate,epsilon=1e-8,name='opt')
    
    
    pred = multilayer_perceptron(weights, biases)
    pred.compile(optimizer=optimizer,loss=cost,metrics=accuracy)
    
   
    first200 = True
    converged = False
    history = pred.fit(xtrain,ytrain, batch_size=batch_size, epochs=200,verbose=0)
    accur[pp][:200] = history.history['loss']
    epoch = 200
    while not converged and epoch < training_epochs:
        history = pred.fit(xtrain,ytrain, batch_size=batch_size, epochs=10,verbose=0)
        accur[pp,epoch:epoch+10] = history.history['loss']
        epoch += 10
        upper = np.mean(accur[pp,epoch-10:epoch])
        middle = np.mean(accur[pp,epoch-20:epoch-10])
        lower = np.mean(accur[pp, epoch-30:epoch-20])
        d1 = 100*abs(upper-middle)/upper
        d2 = 100*abs(middle-lower)/middle
        if d1 < 5 and d2 < 5:
            converged = True
            maxep = epoch
        if epoch == training_epochs:
            converged = True
            print("Warning: Reached max # training epochs:", training_epochs)
            maxep = training_epochs
    
    print('Optimal learning rate: ', learnrate)
    print('Optimal # epochs: ', maxep)
    print('Optimal batch size: ', batch_size)
    
    return(batch_size, learnrate, maxep)     
    



def tf_continue_training(regression, xscale, yscale, x1, y1, ntrees, 
                          learnrate, training_epochs, batch_size,
                          dropout_keep_prob, nhl, display_step, 
                          modelname, modelpath, random_state):
    """ Reloads and continues training an existing DJINN model.
        
    Args:
        xscale (object): Input scaler.
        yscale (object): Output scaler.
        x1 (ndarray): Input features.
        y1 (ndarray): Output features.
        ntrees (int): Number of decision trees.
        learn_rate (float): Learning rate for optimizaiton of weights/biases.
        training_epochs (int): Number of epochs to train neural network.
        batch_size (int): Number of samples per batch.
        dropout_keep_prob (float): Probability of keeping neuron "on" in dropout layers.
        nhl (ndarray): Number of hidden layers in neural network.
        display_step (int): Cost is printed every display_steps during training.
        model_name (str): File name for model if 'save_model' is True.
        model_path (str): Location of where the model/files are saved. 
        random_state (int): Sets random seed.

    Returns:
        None. Re-saves trained model.
    """
    nhl = int(nhl)
    model_path=modelpath
    model_name=modelname
    if(y1.size > y1.shape[0]): n_classes = y1.shape[1]
    else: n_classes = 1

    sess = {}
    xtrain = xscale.transform(x1)
    n_input = xtrain.shape[1]
    
    if regression == True:
        if(n_classes == 1): ytrain = yscale.transform(y1.reshape(-1,1)).flatten()
        else: ytrain = yscale.transform(y1)

    #for classification, do one-hot encoding on classes
    if regression == False:
        ytrain=np.equal.outer(y1.flatten(), np.arange(len(np.unique(y1)))).astype("float32")

    nninfo = {}
    nninfo['weights'] = {}; nninfo['biases'] = {}; nninfo['initial_weights'] = {};
    if (random_state): tf.random.set_seed(random_state)
    for pp in range(0, ntrees):
        pred = tf.keras.models.load_model('%s%s_tree%s.ckpt'%(model_path,model_name,pp),custom_objects={'WB_Init': WB_Init})
        print("Model %s restored"%pp)

        #Restore weights from previous session
        
        weights={}; biases={};
        wb = pred.get_weights()
        for i in range(0,nhl):
            # weights['h%s'%i] = sess[pp].graph.get_tensor_by_name("w%s:0"%i)
            # biases['h%s'%i] = sess[pp].graph.get_tensor_by_name("b%s:0"%i)
            weights['h%s'%(i+1)] = wb[2*i]
            biases['h%s'%(i+1)] = wb[2*i+1]
        # weights['out'] = sess[pp].graph.get_tensor_by_name("w%s:0"%(nhl+1))  
        # biases['out'] = sess[pp].graph.get_tensor_by_name("b%s:0"%nhl)  
        weights['out'] = wb[-2]  
        biases['out'] = wb[-1]  
        nninfo['initial_weights']['tree%s'%pp] = weights

        for layer in pred.layers:
            if isinstance(layer,tf.keras.layers.Dropout):
                layer.rate = 1.0-dropout_keep_prob

        
        history = pred.fit(xtrain,ytrain,batch_size=batch_size,epochs=training_epochs,verbose=2)
        print("Optimization Finished!")

        #save model and nn info 
        save_path = "%s%s_tree%s.ckpt"%(modelpath, modelname, pp)
        pred.save(save_path,save_format='h5')
        print("Model resaved in file: %s" % save_path)
        
        denseWeights = []
        denseBiases = []
        weightsAndBiases = pred.get_weights()
        for i in range(int(len(weightsAndBiases)*.5)):
            denseWeights.append(weightsAndBiases[2*i])
            denseBiases.append(weightsAndBiases[2*i+1])
        nninfo['weights']['tree%s'%pp] = denseWeights
        nninfo['biases']['tree%s'%pp] = denseBiases
        
    with open('%sretrained_nn_info_%s.pkl'%(modelpath, modelname), 'wb') as f1:
        cPickle.dump(nninfo, f1)    
    return()


class WB_Init(tf.keras.initializers.Initializer):
    def __init__(self,dat=None,name=None):
        if isinstance(dat,tf.Tensor):
            self.dat = dat.numpy()
        else:
            self.dat = dat
        self.name = name

                
    def __call__(self,shape,dtype):
        
        if not isinstance(self.dat,tf.Tensor):
            a = tf.convert_to_tensor(self.dat,dtype=tf.float32,name=self.name)
        else:
            a = self.dat
        
        return a
    
    def get_config(self):
        return {'dat': self.dat,
                'name': self.name}
