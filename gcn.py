#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from networkx import karate_club_graph, to_numpy_matrix
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import urllib.request as urllib
import io, os, zipfile


def get_G(community='football'):
    if community == 'karate':
        G = karate_club_graph()
        nodes = sorted(list(G.nodes()))
        A = to_numpy_matrix(G, nodelist=nodes)
        labels = [[1, 0] if G.nodes[i]['club'] == 'Officer' else [0, 1] for i in nodes]
        label_dict = {0:'Officer', 1:'Mr. Hi'}
    elif community == 'football':
        url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"
        sock = urllib.urlopen(url)
        s = io.BytesIO(sock.read())
        sock.close()
        zf = zipfile.ZipFile(s)
        gml = zf.read("football.gml").decode()
        gml = gml.split("\n")[1:]
        G = nx.parse_gml(gml)
        nodes = np.unique(G.nodes)
        L_nodes = len(nodes)
        A_matrix = pd.DataFrame(np.zeros([L_nodes, L_nodes]), columns=nodes, index=nodes)
        for ri, ci in G.edges:
            A_matrix.loc[ri, ci] += 1
        A = (A_matrix + A_matrix.T).values
        Label=[]
        for ni in nodes:
            Label.append(G.nodes[ni]['value'])
        labels = tf.one_hot(Label, np.unique(Label).shape[0])
        label_dict = {}
        for li, ni in zip(Label, nodes):
            label_dict[li] = ni
    else:
        print("No this community.")
    return A, labels, label_dict


class G_Model():
    
    def __init__(self, A, labels, in_units, out_units):
        A = tf.cast(A, tf.float32)
        A_shape = A.shape
        I = tf.eye(A_shape[0])
        AI = A + I
        D = tf.linalg.diag(tf.reduce_sum(AI, axis=0) ** -0.5)
        self.AID = tf.einsum('ij, ik->ik', tf.einsum('ij, jk->ik', D, AI), D)
        self.labels = tf.cast(labels, tf.float16)
        self.feature_dim = in_units
        self.X = tf.Variable(tf.random.truncated_normal([A_shape[1], in_units], \
                                                   mean=0., \
                                                   stddev=1., \
                                                   dtype=tf.float32))
        self.W = tf.Variable(tf.random.truncated_normal([in_units, out_units], \
                                                   mean=0., \
                                                   stddev=1., \
                                                   dtype=tf.float32))
        self.features=[self.X.numpy()]
        self.Loss = []
        self.Accuracy = []
        self.cross_entropy = CategoricalCrossentropy(from_logits=True)
        self.optimizer = Adam(0.01)
        self.pca = PCA()
    
    def generator(self):
        for ai, li in zip(self.AID, self.labels):
            yield ai, li
    
    def forward(self, AID):
        return softmax(tf.einsum('ij, jk->ik', tf.einsum('ij, jk->ik', AID, self.X), self.W))
    
    def __call__(self, loop_unit=32, loops=100, buffer_size=32, batch_size=128):
        print('\ntraining features...')
        for _ in tqdm(range(loops)):
            data = tf.data.Dataset.from_generator(lambda:self.generator(), (tf.float32, tf.float16)).\
                    shuffle(buffer_size).\
                    repeat(loop_unit).\
                    batch(batch_size).\
                    prefetch(tf.data.experimental.AUTOTUNE)
            for aid, label in data:
                with tf.GradientTape() as g:
                    predicts = self.forward(aid)
                    loss = self.cross_entropy(label, predicts)
                gradients = g.gradient(loss, [self.X, self.W])
                self.optimizer.apply_gradients(zip(gradients, [self.X, self.W]))
                accuracy = (label.numpy() == predicts.numpy().round()).mean()
            self.Loss.append(loss.numpy())
            self.Accuracy.append(accuracy)
            self.features.append(self.X.numpy())
        
    def display_features(self, label_dict, save_dir='d:/'):
        def plot1fig(ax, data, xlabel, legend):
            dim = data.shape[1]
            for ix, li in zip(ixs, label_uni):
                if legend:
                    label = label_dict[li]
                else:
                    label = None
                data_ix = data[ix]
                if dim == 1:
                    ax.scatter(data_ix, [0.0] * ix.sum(), linewidths=12, label=label)
                elif dim == 2:
                    ax.scatter(data_ix[:, 0], data_ix[:, 1], linewidths=12, label=label)
                elif dim == 3:
                    ax.scatter(data_ix[:, 0], data_ix[:, 1], data_ix[:, 2], linewidths=12, label=label)
                    ax.set_zticklabels([])
                else:
                    assert dim in [1, 2, 3], "data shape is wrong."
            ax.set_xlabel(xlabel, {'size':30, 'color':'r'})
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if legend:
                ax.legend(bbox_to_anchor=(0, 1.03), loc=3, borderaxespad=0, ncol=4, prop = {'size':36})
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.labels = tf.argmax(self.labels, axis=-1)
        labels = self.labels.numpy().flatten()
        if self.feature_dim > 1:
            features_pca = [self.pca.fit_transform(fi) for fi in self.features]     
        label_uni = set(labels)
        ixs = [labels == li for li in label_uni]
        if self.feature_dim == 1:
            fig = plt.figure(figsize=(16, 12))
        else:
            fig = plt.figure(figsize=(32, 24))
        print('\nplotting features...')
        for i in tqdm(range(len(self.features))):
            fi = self.features[i]
            if self.feature_dim < 4:
                if self.feature_dim == 1:
                    ax = fig.add_subplot(111)
                elif self.feature_dim <3:
                    ax = fig.add_subplot(221)
                else:
                    ax = fig.add_subplot(221, projection='3d')
                plot1fig(ax, fi, str(self.feature_dim) + 'D Feature Representation' + '\nTraining Step: ' + str(i), True)
            if self.feature_dim > 1:
                fi_pca = features_pca[i]
                xlabel_prefix = str(self.feature_dim) + 'D ->'
                for di in range(min(self.feature_dim, 3)):
                    xlabel = xlabel_prefix + str(di+1) +'D Feature Representation by PCA.\nTraining Step: ' + str(i)
                    if di == 2:
                        ax = fig.add_subplot(2, 2, 4-di, projection='3d')
                    else:
                        ax = fig.add_subplot(2, 2, 4-di)
                    if self.feature_dim > 3 and di == 2:
                        plot1fig(ax, fi_pca[:, :di+1], xlabel, True)
                        ax.legend(bbox_to_anchor=(1.0, 1.03), loc=4, borderaxespad=0, ncol=4, prop = {'size':36})
                    else:
                        plot1fig(ax, fi_pca[:, :di+1], xlabel, False)
            plt.savefig(save_dir + str(i) + '.png')
            plt.clf()
        plt.close()

def dim_group_train(base_dir='', dims=[1, 2, 3, 4, 5], out=12, loops=100):
    Ls, As = [], []
    for i in dims:
        model = G_Model(A, labels, i, out)
        model(loops=loops)
        Ls.append(model.Loss)
        As.append(model.Accuracy)
        save_dir = base_dir + str(i) + '/'
        model.display_features(label_dict, save_dir)
    _, axs = plt.subplots(figsize=(12, 5), nrows=1, ncols=2)
    axs = axs.flatten()
    dims = [str(i) for i in dims]
    for axi, lai, ti in zip(axs[::-1], [As, Ls], ['Accuracy', 'Loss']):
        for lai_i, di in zip(lai, dims):
            axi.plot(lai_i, label=di)
            axi.grid(linestyle='--') 
        axi.set_xlabel(ti, color='r')
    axi.legend(bbox_to_anchor=(0.0, 1.03), loc=3, borderaxespad=0, ncol=len(dims), prop = {'size':16})
    plt.savefig(base_dir + 'loss_accuracy.png')
    plt.close()


A, labels, label_dict = get_G('football')
dim_group_train(base_dir='../football/', dims=[1, 2, 3, 4, 5], out=12, loops=50)




