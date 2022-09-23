# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 15:02:47 2021

@author: Wanlun Ma
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sklearn
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt



# @tf.function
def two_decom_cov(r, y, max_iter=1000, diff_threshold=1e-5,balance_sample=True):
    class_num, labels = tf.unique(y)
    sample_num_class = (tf.where(labels == tf.cast(class_num[0], dtype=tf.int32))).shape[0]
    # print("sample_num_class:",sample_num_class,"len(class_num):",len(class_num))
    index_list = []
    u_start = []
    e_start = []
    u_c_start = []
    for i in range(len(class_num)):
        index = tf.where(labels == tf.cast(class_num[i], dtype=tf.int32))
        index_list.append(index)
        print(f'class:{i},sample_num:{index.shape[0]}')
        r_c = tf.gather_nd(r, indices=index)
        r_c_mean = tf.reduce_mean(r_c,axis=0,keepdims=True)
        u_c_start.append(r_c_mean)
        u_start.append(tf.tile(r_c_mean,(index.shape[0],1)))
        e_start.append(r_c - r_c_mean)
    if u_start[0].shape[-1] < 100:
        print(f'---u_start:\n{tf.concat(u_c_start,axis=0).numpy()}')
    iter_i = 0
    diff = tf.Variable(1.,dtype=tf.float32)
    u_class = tf.Variable(tf.concat(u_c_start,axis=0),dtype=tf.float32)
    u = tf.Variable(tf.concat(u_start,axis=0),dtype=tf.float32)
    res = tf.Variable(tf.concat(e_start,axis=0),dtype=tf.float32)
    S_u = tf.Variable(tfp.stats.covariance(u, sample_axis=0, event_axis=-1),dtype=tf.float32)
    S_e = tf.Variable(tfp.stats.covariance(res, sample_axis=0, event_axis=-1),dtype=tf.float32)
    while iter_i < max_iter and diff > diff_threshold:
        iter_i = iter_i + 1
        if iter_i % (max_iter//5) ==0:
            print(f'iter_i={iter_i},    S_e diff={diff.numpy()}')
            # print(f'S_e={S_e.numpy()},S_u={S_u.numpy()}')
        #### E-step
        F = tf.linalg.pinv(S_e)
        if balance_sample == True:
            G = -tf.linalg.matmul(tf.linalg.matmul(tf.linalg.pinv(sample_num_class*S_u+S_e),S_u),F)
            u_all = -tf.linalg.matmul(tf.linalg.matmul(S_e,G),r,transpose_b=True)
            # u_all1 = tf.linalg.matmul(tf.linalg.matmul(S_u, (F + sample_num_class * G)), r, transpose_b=True)
            # print(f'error of u from different G:{tf.norm(u_all-u_all1)}')
            u_list = []
            e_list = []
            u_c_list = []
            for i in range(len(class_num)):
                index = index_list[i]
                u_c = tf.reduce_sum(tf.gather_nd(tf.transpose(u_all), indices=index),axis=0,keepdims=True)
                u_c_list.append(u_c)
                u_list.append(tf.tile(u_c,(sample_num_class,1)))
                e_list.append(tf.gather_nd(r, indices=index) - u_list[i])
        else:
            u_list = []
            e_list = []
            u_c_list = []
            for i in range(len(class_num)):
                index = index_list[i]
                r_c = tf.gather_nd(r, indices=index)
                G = -tf.linalg.matmul(tf.linalg.matmul(tf.linalg.pinv(index.shape[0] * S_u + S_e), S_u), F)
                #u_c = tf.linalg.matmul(tf.linalg.matmul(S_u, (F + index.shape[0] * G)), r_c, transpose_b=True)
                u_c = -tf.linalg.matmul(tf.linalg.matmul(S_e,G),r_c,transpose_b=True)
                u_c = tf.reduce_sum(tf.transpose(u_c), axis=0,keepdims=True)
                u_c_list.append(u_c)
                u_list.append(tf.tile(u_c,(index.shape[0],1)))
                e_list.append(r_c - u_list[i])

        u_class.assign(tf.concat(u_c_list,axis=0))
        u.assign(tf.concat(u_list,axis=0))
        res.assign(tf.concat(e_list,axis=0))
        #res1 = r + tf.reduce_sum(tf.linalg.matmul(tf.linalg.matmul(S_e,G),r,transpose_b=True),axis=1)

        ##### M-step
        # S_e_new = np.cov(e_list.numpy(), rowvar=False)
        # S_u_new = np.cov(u_list.numpy(), rowvar=False)
        S_e_new = tfp.stats.covariance(res, sample_axis=0, event_axis=-1)
        S_u_new = tfp.stats.covariance(u, sample_axis=0, event_axis=-1)
        diff = tf.norm(S_e_new - S_e) / tf.norm(S_e_new)
        S_e = tf.cast(S_e_new,dtype=tf.float32)
        S_u = tf.cast(S_u_new,dtype=tf.float32)

    print(f'COV stop at: iter_i={iter_i}, S_e diff={diff.numpy()}')
    # print(S_e, S_u)
    return u_class.numpy(), res.numpy(), S_u.numpy(), S_e.numpy(),

# @tf.function
def two_decom(r, y, S_u, S_e):
    class_num, labels = tf.unique(y)
    sample_num_class = (tf.where(labels == tf.cast(class_num[0], dtype=tf.int32))).shape[0]
    print("sample_num_class:",sample_num_class)
    F = tf.linalg.pinv(S_e)
    u_list = []
    e_list = []
    for i in range(len(class_num)):
        index = tf.where(labels == tf.cast(class_num[i], dtype=tf.int32))
        r_c = tf.gather_nd(r, indices=index)
        # r_c = r_c - tf.reduce_mean(r_c,axis=0,keepdims=True) ## centralize in class
        G = -tf.linalg.matmul(tf.linalg.matmul(tf.linalg.pinv(index.shape[0] * S_u + S_e), S_u), F)
        #u_c = tf.linalg.matmul(tf.linalg.matmul(S_u, (F + index.shape[0] * G)), r_c, transpose_b=True)
        u_c = -tf.linalg.matmul(tf.linalg.matmul(S_e, G), r_c, transpose_b=True)
        u_c_EM = tf.reduce_sum(tf.transpose(u_c), axis=0,keepdims=True)
        # print("u_c_EM:",u_c_EM.shape)
        u_list.append(u_c_EM)
        e_list.append(r_c - u_c_EM)

    u = tf.concat(u_list,axis=0)
    res = tf.concat(e_list,axis=0)
    return u.numpy(), res.numpy()

# @tf.function
def two_sub(r_c,y_sub,S_e,S_u,max_iter=1000):
    #y_sub = np.argmax(y_sub, axis=-1)
    F = tf.linalg.pinv(S_e)
    iter_i = 0
    diff = tf.Variable(4.,dtype=tf.float32)
    diff1 = 1.
    u1_old = tf.expand_dims(r_c[0], axis=1)
    u1 = tf.Variable(tf.expand_dims(r_c[0], axis=1), dtype=tf.float32)
    u2 = tf.Variable(tf.expand_dims(r_c[0], axis=1), dtype=tf.float32)
    # print("y_sub.shape",y_sub.shape,"u1.shape",u1.shape)
    y_sub = tf.cast(y_sub,dtype=tf.int32)
    diff_unchange = 0.
    Restart_count = 0
    while iter_i < max_iter and (diff > 1 or diff1 > 1e-5) and diff_unchange < 3:
        iter_i = iter_i + 1
        diff_old = diff
        if iter_i % 1000 == 0:
            print(f'subgroup: iter_i={iter_i},    diff={diff.numpy()},    diff1={diff1.numpy()},  diff_unchange={diff_unchange}')
        
        v = tf.cast(tf.linalg.matmul(F, (u1 - u2)),dtype=tf.float32)
        u1_projected = tf.linalg.matmul(tf.linalg.matmul(u1, F, transpose_a=True), u1)
        u2_projected = tf.linalg.matmul(tf.linalg.matmul(u2, F, transpose_a=True), u2)
        t = 0.5 * (u1_projected - u2_projected)
        r_projected = tf.linalg.matmul(v, r_c, transpose_a=True, transpose_b=True) - t
        c = tf.where(r_projected > 0, 0, 1)
        diff = tf.cast(tf.reduce_sum(tf.abs(y_sub - c)),dtype=tf.float32)
        y_sub = tf.squeeze(c)

        class_num, labels = tf.unique(y_sub)
        if len(class_num) < 2:
            if iter_i > 0:
                Restart_count = Restart_count +1
                if Restart_count % int(max_iter*0.1) ==0:
                    print(f"Restart count:{Restart_count}")
            y_sub = tf.convert_to_tensor(np.random.randint(low=0, high=2, size=r_c.shape[0]),dtype=tf.int32)
            class_num, labels = tf.unique(y_sub)

        u_list = []
        for i in range(len(class_num)):
            index = tf.where(labels == tf.cast(class_num[i], dtype=tf.int32))
            r_c_sub = tf.gather_nd(r_c, indices=index)
            # print("class:",i,"index:",index.shape,"r_c_sub:",r_c_sub.shape)
            G = -tf.linalg.matmul(tf.linalg.matmul(tf.linalg.pinv(index.shape[0] * S_u + S_e), S_u), F)
            # u_sub = tf.linalg.matmul(tf.linalg.matmul(S_u, (F + index.shape[0] * G)), r_c_sub, transpose_b=True)
            u_sub = -tf.linalg.matmul(tf.linalg.matmul(S_e, G), r_c_sub, transpose_b=True)
            u_sub_EM = tf.reduce_sum(u_sub, axis=1)
            # r_c_sub_mean = tf.reduce_mean(r_c_sub,axis=0)
            # print("u error using EM:",tf.norm(u_sub_EM-r_c_sub_mean).numpy())
            u_list.append(u_sub_EM)

        u1.assign(tf.expand_dims(u_list[0], axis=1))
        u2.assign(tf.expand_dims(u_list[1], axis=1))

        diff1 = tf.norm(u1 - u1_old) / tf.norm(u1)
        u1_old = u1

        if tf.abs(diff-diff_old)< 1.:
            # print("diff_unchange:",diff_unchange)
            diff_unchange = diff_unchange + 1.
        else:
            diff_unchange = 0.
    if r_c.shape[-1]<100:
        print(f'u1:{np.transpose(u1.numpy())}, \nu2:{np.transpose(u2.numpy())}')
    u_sub = tf.transpose(tf.where(y_sub < 1, u1, u2))
    # u_sub = tf.transpose(tf.where(y_sub > 0, u1, u2))
    print(f'untangling stop at: iter_i={iter_i},    diff={diff.numpy()},    diff1={diff1.numpy()},  diff_unchange={diff_unchange},  Restart_count={Restart_count}')
    print(f'y_sub={y_sub[:500]},{y_sub[500:]}')
    # print(u1,u2)
    # print(u_sub)
    return u_sub.numpy(), y_sub.numpy(), r_projected.numpy()

def poison_img(images,offset = 1):
    mask = np.zeros(shape=images[0].shape)

    h, w = mask.shape[0] // 2 +offset, mask.shape[1] // 2 +offset
    mask[h, w] = (1., 1., 1.)
    mask[h + 1, w] = (1., 1., 1.)
    mask[h, w + 1] = (1., 1., 1.)
    mask[h + 1, w + 1] = (1., 1., 1.)

    # mask[h - 1, w] = (1., 1., 1.)
    # mask[h, w - 1] = (1., 1., 1.)
    # mask[h - 1, w - 1] = (1., 1., 1.)
    # mask[h + 1, w - 1] = (1., 1., 1.)
    # mask[h - 1, w + 1] = (1., 1., 1.)

    trigger = np.ones(shape=images[0].shape) * 1 * mask
    poison_images = images * (1-np.expand_dims(mask,axis=0)) + np.expand_dims(trigger,axis=0)

    plt.imshow(trigger)
    path = './log/trigger_solid_md2'
    plt.imsave(f'{path}/mask_offset={offset}.png', trigger)
    # plt.show()
    return poison_images

def TaCT(X_all, y_all, n_poison=1000, n_cover=1000, target_class=None, source_class=None, cover_class=None):
    if target_class is None:
        target_class = [0]
    if source_class is None:
        source_class = [1]
    if cover_class is None:
        cover_class = [2, 3, 4, 5, 6, 7, 8, 9]
    print(f'target_class:{target_class},source_class:{source_class},cover_class:{cover_class}')
    labels = np.argmax(y_all, axis=-1)
    class_num = np.unique(labels)
    train_classes = np.arange(len(class_num))
    print("len(class_num):",len(class_num))
    target_class = np.array(target_class)
    source_class = np.array(source_class)
    cover_class = np.array(cover_class) # np.array([2, 3])#
    n_cover = n_cover//len(cover_class)
    clean_x = []
    clean_y = []
    contaminant_x = []
    contaminant_y = []

    for tc in train_classes:
        index = np.where(labels == tc)
        if tc in target_class:
            clean_x.append(X_all[index])
            clean_y.append(y_all[index])
        elif tc in source_class:
            clean_x.append(X_all[index][n_poison*len(target_class):])
            clean_y.append(y_all[index][n_poison*len(target_class):])
            contaminant_x.append(X_all[index][:n_poison*len(target_class)])
            for tarc in target_class:
                y_tarc = np.expand_dims(y_all[np.where(labels == tarc)][0],0)
                contaminant_y.append(np.tile(y_tarc,[n_poison,1]))
        elif tc in cover_class:
            clean_x.append(X_all[index][n_cover:])
            clean_y.append(y_all[index][n_cover:])
            contaminant_x.append(X_all[index][:n_cover])
            contaminant_y.append(y_all[index][:n_cover])
        else:
            clean_x.append(X_all[index])
            clean_y.append(y_all[index])

    clean_x = np.concatenate(clean_x, axis=0)
    contaminant_x = poison_img(np.concatenate(contaminant_x, axis=0))
    train_x = np.concatenate([clean_x, contaminant_x], axis=0)

    clean_y = np.concatenate(clean_y, axis=0)
    contaminant_y = np.concatenate(contaminant_y, axis=0)
    train_y = np.concatenate([clean_y,contaminant_y], axis=0)
    print(f'clean_x:{len(clean_x)},clean_y:{len(clean_y)},contaminant_x:{len(contaminant_x)},contaminant_y:{len(contaminant_y)}')
    print('data loaded, training x:{0}, y:{1}; '.format(train_x.shape,train_y.shape))
    return train_x, train_y

def TaCT_v2(X_all, y_all, n_poison=1000, n_cover=1000, target_class=None, source_class=None, cover_class=None):
    if target_class is None:
        target_class = [0]
    if source_class is None:
        source_class = [1]
    if cover_class is None:
        cover_class = [2, 3, 4, 5, 6, 7, 8, 9]
    print(f'target_class:{target_class},source_class:{source_class},cover_class:{cover_class}')
    labels = np.argmax(y_all, axis=-1)
    class_num = np.unique(labels)
    train_classes = np.arange(len(class_num))
    print("len(class_num):",len(class_num))
    target_class = np.array(target_class)
    source_class = np.array(source_class)
    cover_class = np.array(cover_class) # np.array([2, 3])#
    n_cover = n_cover//len(cover_class)
    clean_x = []
    clean_y = []
    contaminant_x = []
    contaminant_y = []

    for tc in train_classes:
        index = np.where(labels == tc)
        if tc in target_class:
            clean_x.append(X_all[index])
            clean_y.append(y_all[index])
        elif tc in source_class:
            clean_x.append(X_all[index])
            clean_y.append(y_all[index])
            contaminant_x.append(X_all[index][:n_poison*len(target_class)])
            for tarc in target_class:
                y_tarc = np.expand_dims(y_all[np.where(labels == tarc)][0],0)
                contaminant_y.append(np.tile(y_tarc,[n_poison,1]))
        elif tc in cover_class:
            clean_x.append(X_all[index])
            clean_y.append(y_all[index])
            contaminant_x.append(X_all[index][:n_cover])
            contaminant_y.append(y_all[index][:n_cover])
        else:
            clean_x.append(X_all[index])
            clean_y.append(y_all[index])

    clean_x = np.concatenate(clean_x, axis=0)
    contaminant_x = poison_img(np.concatenate(contaminant_x, axis=0))
    train_x = np.concatenate([clean_x, contaminant_x], axis=0)

    clean_y = np.concatenate(clean_y, axis=0)
    contaminant_y = np.concatenate(contaminant_y, axis=0)
    train_y = np.concatenate([clean_y,contaminant_y], axis=0)
    print(f'clean_x:{len(clean_x)},clean_y:{len(clean_y)},contaminant_x:{len(contaminant_x)},contaminant_y:{len(contaminant_y)}')
    print('data loaded, training x:{0}, y:{1}; '.format(train_x.shape,train_y.shape))
    return train_x, train_y


def clean_dataset(X_all,y_all,n_clean=100,n_clean_test=100,num_class=10):
    #X_all,y_all = sklearn.utils.shuffle(X_all,y_all)
    if len(y_all.shape) >1:
        labels = np.argmax(y_all, axis=-1)
    else:
        labels = y_all
    train_classes = np.arange(num_class)
    clean_x = []
    clean_y = []
    clean_x_test = []
    clean_y_test = []
    for tc in train_classes:
        index = np.where(labels == tc)
        clean_x.append(X_all[index][0:n_clean])
        clean_y.append(y_all[index][0:n_clean])
        clean_x_test.append(X_all[index][-n_clean_test:])
        clean_y_test.append(y_all[index][-n_clean_test:])
    clean_x = np.concatenate(clean_x, axis=0)
    clean_y = np.concatenate(clean_y, axis=0)
    clean_x_test = np.concatenate(clean_x_test, axis=0)
    clean_y_test = np.concatenate(clean_y_test, axis=0)
    return clean_x, clean_y, clean_x_test, clean_y_test

def bd_dataset(X_all,y_all,y_all_bd,n_poison=100,num_class=10,target_class=[],source_class=[1],balance=True):
    #X_all,y_all = sklearn.utils.shuffle(X_all,y_all)
    if len(y_all.shape) >1:
        print('y_all.shape:',y_all.shape)
        labels = np.argmax(y_all, axis=-1)
    else:
        labels = y_all
    train_classes = np.arange(num_class)
    clean_x = []
    clean_y = []
    poison_y = []
    if balance:
        for tc in train_classes: ## poison mode
            if tc in target_class:
                continue
            if tc not in source_class:
                continue
            index = np.where(labels == tc)
            clean_x.append(X_all[index][0:n_poison])
            clean_y.append(y_all[index][0:n_poison])
            # poison_index = np.where(labels == target_class[0])
            # poison_y.append(y_all[poison_index][0:n_poison])
            poison_y.append(y_all_bd[index][0:n_poison])
        clean_x = np.concatenate(clean_x, axis=0)
        clean_y = np.concatenate(clean_y, axis=0)
        poison_y = np.concatenate(poison_y, axis=0)
    else:
        index = np.where(labels != target_class[0])
        clean_x.append(X_all[index][0:n_poison])
        clean_y.append(y_all[index][0:n_poison])
        # poison_index = np.where(labels == target_class[0])
        # poison_y.append(y_all[poison_index][0:n_poison])
        poison_y.append(y_all_bd[index][0:n_poison])
        clean_x = np.concatenate(clean_x, axis=0)
        clean_y = np.concatenate(clean_y, axis=0)
        poison_y = np.concatenate(poison_y, axis=0)

    return clean_x, clean_y, poison_y


def clean_dataset_noise(X_all,y_all,y_all_noise,n_clean=100,num_class=10):
    #X_all,y_all = sklearn.utils.shuffle(X_all,y_all)
    if len(y_all.shape) >1:
        labels = np.argmax(y_all, axis=-1)
    else:
        labels = y_all
    train_classes = np.arange(num_class)
    clean_x = []
    clean_y = []
    noise_y = []
    for tc in train_classes:
        index = np.where(labels == tc)
        clean_x.append(X_all[index][-n_clean:])
        clean_y.append(y_all[index][-n_clean:])
        noise_y.append(y_all_noise[index][-n_clean:])
    clean_x = np.concatenate(clean_x, axis=0)
    clean_y = np.concatenate(clean_y, axis=0)
    noise_y = np.concatenate(noise_y, axis=0)
    return clean_x, clean_y, noise_y

def bd_dataset_noise(X_all,y_all,y_all_bd,y_all_noise,n_poison=100,num_class=10,target_class=[],source_class=[1],balance=True):
    #X_all,y_all = sklearn.utils.shuffle(X_all,y_all)
    if len(y_all.shape) >1:
        labels = np.argmax(y_all, axis=-1)
    else:
        labels = y_all
    train_classes = np.arange(num_class)
    clean_x = []
    clean_y = []
    poison_y = []
    noise_y = []
    if balance:
        for tc in train_classes: ## poison mode
            if tc in target_class:
                continue
            if tc not in source_class:
                continue
            index = np.where(labels == tc)
            clean_x.append(X_all[index][0:n_poison])
            clean_y.append(y_all[index][0:n_poison])
            noise_y.append(y_all_noise[index][0:n_poison])
            poison_y.append(y_all_bd[index][0:n_poison])
        clean_x = np.concatenate(clean_x, axis=0)
        clean_y = np.concatenate(clean_y, axis=0)
        noise_y = np.concatenate(noise_y, axis=0)
        poison_y = np.concatenate(poison_y, axis=0)
    else:
        index = np.where(labels != target_class[0])
        clean_x.append(X_all[index][0:n_poison])
        clean_y.append(y_all[index][0:n_poison])
        noise_y.append(y_all_noise[index][0:n_poison])
        poison_y.append(y_all_bd[index][0:n_poison])
        clean_x = np.concatenate(clean_x, axis=0)
        clean_y = np.concatenate(clean_y, axis=0)
        noise_y = np.concatenate(noise_y, axis=0)
        poison_y = np.concatenate(poison_y, axis=0)

    return clean_x, poison_y, noise_y, clean_y

'''
def sudo_data():
    tfd = tfp.distributions
    mean = tf.Variable(tf.zeros([laten_dim]),dtype=tf.float32)
    cov = tf.Variable(2. * tf.eye(laten_dim), dtype=tf.float32)
    scale = tf.linalg.cholesky(cov)
    mvn = tfd.MultivariateNormalTriL(loc=mean, scale_tril=scale)
    
    cov_e = tf.Variable(0.1 * tf.eye(laten_dim), dtype=tf.float32)
    scale = tf.linalg.cholesky(cov_e)
    mvn_e = tfd.MultivariateNormalTriL(loc=mean, scale_tril=scale)
    
    u = mvn.sample(10)  # sampling Gaussian noise
    
    sample_num_class_clean = int(sample_num_class*0.1)
    r_clean = tf.reshape(tf.tile(u,[1,sample_num_class_clean]),(-1,laten_dim)) + mvn_e.sample(10*sample_num_class_clean)
    y_clean = np.array([[0,1,2,3,4,5,6,7,8,9]])
    y_clean = tf.tile(y_clean,[sample_num_class_clean,1])
    y_clean = tf.reshape(tf.transpose(y_clean),(-1))
    
    e = mvn_e.sample(10*sample_num_class)  # sampling Gaussian noise
    r = tf.reshape(tf.tile(u,[1,sample_num_class]),(-1,laten_dim)) + e
    u_extra = mvn.sample(1)
    poison = r[0:100] - (u[0] - tf.squeeze(u_extra))
    r_poison = tf.concat([poison,r[100:]],axis=0)
    y = np.array([np.arange(10)])
    y = tf.tile(y,[sample_num_class,1])
    y = tf.reshape(tf.transpose(y),(-1))
'''

def ASRate(model_clean,model_infect,X_clean,true_label=0,target_label=0):
    x_poison = poison_img(X_clean)
    y_clean = np.argmax(model_infect.predict(X_clean),axis=-1)
    y_poison = np.argmax(model_infect.predict(x_poison),axis=-1)
    attack_success = 0
    correct_sample = 0
    for i in range(X_clean.shape[0]):
        if np.abs(y_clean[i]-true_label)<0.5:
            correct_sample = correct_sample + 1
            if np.abs(y_poison[i]-target_label)<0.5:
                attack_success = attack_success + 1
    return attack_success,correct_sample

