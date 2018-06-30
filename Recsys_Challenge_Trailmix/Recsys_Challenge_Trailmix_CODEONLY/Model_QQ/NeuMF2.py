'''
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.
@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np
import random
import gpu_parallel
import json

#from keras import initializations
from keras.regularizers import l1, l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluation2 import evaluate_model
from Dataset2 import Dataset
from time import time
import sys
#import GMF, MLP
import argparse

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()

# def init_normal(shape, name=None):
#     return initializations.normal(shape, scale=0.01, name=name)

def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                  embeddings_initializer = 'he_normal', embeddings_regularizer = l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                  embeddings_initializer = 'he_normal', embeddings_regularizer = l2(reg_mf), input_length=1)

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = layers[0]/2, name = "mlp_embedding_user",
                                   embeddings_initializer='he_normal', embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = layers[0]/2, name = 'mlp_embedding_item',
                                   embeddings_initializer='he_normal', embeddings_regularizer = l2(reg_layers[0]), input_length=1)

    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = merge([mf_user_latent, mf_item_latent], mode = 'mul') # element-wise multiply

    # MLP part
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode = 'concat')
    for idx in xrange(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer = l2(reg_layers[idx]), activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    #mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    #mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    predict_vector = merge([mf_vector, mlp_vector], mode = 'concat')

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = "prediction")(predict_vector)

    model = Model(input=[user_input, item_input],
                  output=prediction)

    return model

def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)

    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)

    # MLP layers
    for i in xrange(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)

    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])
    return model

def get_train_instances(train, num_negatives, all_item_idices):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    # q0 = np.load('prob100.npy')
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in xrange(num_negatives):
            # j = all_item_idices[np.random.multinomial(1, q0).argmax()]
            j = random.sample(all_item_idices, 1)[0]
            # j = np.random.randint(num_items)
            while train.has_key((u, j)):
                # j = all_item_idices[np.random.multinomial(1, q0).argmax()]
                j = random.sample(all_item_idices, 1)[0]
                # j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


import os
import GPUtil

def select_gpu():
   try:
       # Get the first available GPU
       DEVICE_ID_LIST = GPUtil.getFirstAvailable()
       DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list

       # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
       os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
   except EnvironmentError:
       print("GPU not found")



# config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
# config.gpu_options.allow_growth = True
# sess = tf.InteractiveSession(config=config)
# init = tf.global_variables_initializer()

if __name__ == '__main__':

    select_gpu()

    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain

    # topK = 10
    evaluation_threads = 1  #mp.cpu_count()
    print("NeuMF arguments: %s " %(args))
    model_out_file = 'Pretrain/Inf/Final/%d_%s_NeuMF_%d_%s_%d.h5' %(args.epochs, args.dataset, mf_dim, args.layers, time())
    print("QQ")
    # Loading data
    t1 = time()

    dataset = Dataset()

    # train = dataset.trainMatrix

    all_item_idices = np.load('item_idx_100_up.npy')
    # all_item_idices = np.load('item_idx_5_up.npy')

    num_users, num_items = dataset.trainMatrix.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d"
          %(time()-t1, num_users, num_items, dataset.trainMatrix.nnz))

    # Build model
    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

    # Init performance
    if args.out > 0:
        model.save_weights(model_out_file, overwrite=True)

    # Training model
    for epoch in xrange(num_epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(dataset.trainMatrix, num_negatives, all_item_idices)

        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)],  #input
                         np.array(labels), # labels
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:

            loss = hist.history['loss'][0]
            print('Iteration %d [%.1f s]: loss = %.4f [%.1f s]'
                 % (epoch,  t2-t1, loss, time()-t2))

            if args.out > 0:
                model.save_weights(model_out_file, overwrite=True)

    if args.out > 0:
        print("The best NeuMF model is saved to %s" %(model_out_file))

    # Final = evaluate_model(model, test_list_idices, all_item_idices, testRemove)
    # ff = 'Pretrain/Final/'+str(args.epochs)+'QQ_submit_1.json'
    # with open(ff, 'w') as fp:
    #      json.dump(Final, fp)
    #
    # # Change to orginal
    # b = np.load('song_name.py.npy')
    # k = 0
    # for p in Final.keys():
    #     k += 1
    #     if k % 10000 == 0:
    #         print(k)
    #     for q in range(len(Final[p])):
    #         Final[p][q] = b[Final[p][q]]
    #
    # ff = 'Pretrain/Final/'+str(args.epochs)+'QQ_submit_2.json'
    # with open(ff, 'w') as fp:
    #      json.dump(Final, fp)
