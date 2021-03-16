import tensorflow as tf
from tensorflow import keras
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For multiple devices (GPUs: 4, 5, 6, 7)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,4,5,6,7"
# import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import time, sys
import pickle
import timeit
from scipy.interpolate import make_interp_spline, BSpline
import wandb
#os.environ["WANDB_API_KEY"] = "key_code"
plt.ioff()
cifar10 = tf.keras.datasets.cifar10
# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), progress * 100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


class Deterministic_Conv(keras.layers.Layer):
    def __init__(self, kernel_size=5, kernel_num=16, kernel_stride=1, padding="VALID"):
        super(Deterministic_Conv, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.kernel_stride = kernel_stride
        self.padding = padding
    def build(self, input_shape):
        self.w = self.add_weight(name='w',
                                 shape=(self.kernel_size, self.kernel_size, input_shape[-1], self.kernel_num),
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None),                       
                                 trainable=True,               )
    def call(self, input_in):        
        out = tf.nn.conv2d(input_in, self.w, strides=[1, self.kernel_stride, self.kernel_stride, 1],
                              padding=self.padding, data_format='NHWC')        
        return out

class DMaxPooling(keras.layers.Layer):    
    def __init__(self, pooling_size=2, pooling_stride=2, pooling_pad='SAME'):
        super(DMaxPooling, self).__init__()
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.pooling_pad = pooling_pad
    def call(self, input_in):        
        out, argmax_out = tf.nn.max_pool_with_argmax(input_in, ksize=[1, self.pooling_size, self.pooling_size, 1],
                                                        strides=[1, self.pooling_stride, self.pooling_stride, 1],
                                                        padding=self.pooling_pad)  # shape=[batch_zise, new_size,new_size,num_channel]  
        return out

class DFlatten_and_FC(keras.layers.Layer):   
    def __init__(self, units):
        super(DFlatten_and_FC, self).__init__()
        self.units = units                
    def build(self, input_shape):
        self.w = self.add_weight(name = 'w', shape=(input_shape[1]*input_shape[2]*input_shape[-1], self.units),
            initializer=tf.random_normal_initializer( mean=0.0, stddev=0.05, seed=None), 
            trainable=True,
        )          
    def call(self, input_in):
        batch_size = input_in.shape[0]           
        flatt = tf.reshape(input_in, [batch_size, -1]) #shape=[batch_size, im_size*im_size*num_channel]           
        out = tf.matmul(flatt, self.w)        
        return out  

class Dsoftmax(keras.layers.Layer):
    def __init__(self):
        super(Dsoftmax, self).__init__()
    def call(self, input_in):
        out = tf.nn.softmax(input_in)        
        return out

class DReLU(keras.layers.Layer):
    def __init__(self):
        super(DReLU, self).__init__()
    def call(self, input_in):
        out = tf.nn.relu(input_in)        
        return out

class DELU(keras.layers.Layer):
    def __init__(self):
        super(DELU, self).__init__()
    def call(self, input_in):
        out = tf.nn.elu(input_in)       
        return out

class DDropout(keras.layers.Layer):
    def __init__(self, drop_prop):
        super(DDropout, self).__init__()
        self.drop_prop = drop_prop
    def call(self, input_in, Training=True):               
        if Training:        
           out = tf.nn.dropout(input_in, rate=self.drop_prop)           
        else:
           out = input_in                
        return out

class DBatch_Normalization(keras.layers.Layer):
    def __init__(self, var_epsilon):
        super(DBatch_Normalization, self).__init__()
        self.var_epsilon = var_epsilon
    def call(self, input_in):
        mean, variance = tf.nn.moments(input_in, [0, 1, 2])
        out = tf.nn.batch_normalization(input_in, mean, variance, offset=None, scale=None, variance_epsilon=self.var_epsilon)        
        return out        
        
class Deterministic_CNN(tf.keras.Model):
    def __init__(self, kernel_size, num_kernel, pooling_size, pooling_stride, pooling_pad, units, drop_prop=0.2,
                  var_epsilon=1e-4, name=None):#
        super(Deterministic_CNN, self).__init__()
        self.kernel_size = kernel_size
        self.num_kernel = num_kernel
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.pooling_pad = pooling_pad
        self.units = units
        self.drop_prop = drop_prop
        self.var_epsilon = var_epsilon

        self.conv_1 = Deterministic_Conv(kernel_size=self.kernel_size[0], kernel_num=self.num_kernel[0], padding='VALID')
        self.conv_2 = Deterministic_Conv(kernel_size=self.kernel_size[1], kernel_num=self.num_kernel[1], padding='SAME')
        self.conv_3 = Deterministic_Conv(kernel_size=self.kernel_size[2], kernel_num=self.num_kernel[2], padding='SAME')
        self.conv_4 = Deterministic_Conv(kernel_size=self.kernel_size[3], kernel_num=self.num_kernel[3], padding='SAME')
        self.conv_5 = Deterministic_Conv(kernel_size=self.kernel_size[4], kernel_num=self.num_kernel[4], padding='SAME')
        self.conv_6 = Deterministic_Conv(kernel_size=self.kernel_size[5], kernel_num=self.num_kernel[5], padding='SAME')
        self.conv_7 = Deterministic_Conv(kernel_size=self.kernel_size[6], kernel_num=self.num_kernel[6], padding='SAME')
        self.conv_8 = Deterministic_Conv(kernel_size=self.kernel_size[7], kernel_num=self.num_kernel[7], padding='SAME')
        self.conv_9 = Deterministic_Conv(kernel_size=self.kernel_size[8], kernel_num=self.num_kernel[8], padding='SAME')
        self.conv_10 = Deterministic_Conv(kernel_size=self.kernel_size[9],kernel_num=self.num_kernel[9], padding='SAME')

        self.elu_1 = DELU()
        self.maxpooling_1 = DMaxPooling(pooling_size=self.pooling_size[0], pooling_stride=self.pooling_stride[0],   pooling_pad=self.pooling_pad)
        self.maxpooling_2 = DMaxPooling(pooling_size=self.pooling_size[1], pooling_stride=self.pooling_stride[1],   pooling_pad=self.pooling_pad)
        self.maxpooling_3 = DMaxPooling(pooling_size=self.pooling_size[2], pooling_stride=self.pooling_stride[2],   pooling_pad=self.pooling_pad)
        self.maxpooling_4 = DMaxPooling(pooling_size=self.pooling_size[3], pooling_stride=self.pooling_stride[3],   pooling_pad=self.pooling_pad)
        self.maxpooling_5 = DMaxPooling(pooling_size=self.pooling_size[4], pooling_stride=self.pooling_stride[4],   pooling_pad=self.pooling_pad)

        self.dropout_1 = DDropout(self.drop_prop)        
        self.batch_norm = DBatch_Normalization(self.var_epsilon)
        self.fc_1 = DFlatten_and_FC(self.units)
        self.mysoftma = Dsoftmax()

    def call(self, inputs, training=True):
        out = self.conv_1(inputs)       
        out = self.elu_1(out)
        out = self.batch_norm(out)
        out = self.maxpooling_1(out) 
      
        out = self.conv_2(out)        
        out = self.elu_1(out)
        out = self.batch_norm(out)
        out = self.conv_3(out)        
        out = self.elu_1(out)
        out = self.batch_norm(out)
        out = self.maxpooling_2(out) 
        out = self.dropout_1(out,  Training=training)           
        
        out = self.conv_4(out)       
        out = self.elu_1(out) 
        out = self.batch_norm(out)
        out = self.conv_5(out)       
        out = self.elu_1(out)
        out = self.batch_norm(out)
        out = self.maxpooling_3(out)
        out = self.dropout_1(out,  Training=training)   
        
        out = self.conv_6(out)      
        out = self.elu_1(out)
        out = self.batch_norm(out)
        out = self.conv_7(out)      
        out = self.elu_1(out) 
        out = self.batch_norm(out)
        out = self.maxpooling_4(out)
        out = self.dropout_1(out,  Training=training)   
              
        out = self.conv_8(out)      
        out = self.elu_1(out)
        out = self.batch_norm(out)
        out = self.conv_9(out)      
        out = self.elu_1(out)
        out = self.batch_norm(out)
        out = self.maxpooling_5(out)
        out = self.dropout_1(out,  Training=training)   
        
        out = self.conv_10(out)       
        out = self.elu_1(out)
        out = self.batch_norm(out)                
        out = self.fc_1(out)
        output = self.mysoftma(out)        
        return output

def main_function(input_dim=32,n_channels=3, num_kernels=[32, 32, 32, 32, 64, 64, 64, 128, 128, 128],
                  kernels_size=[5, 3, 3, 3, 3, 3, 3, 3, 3, 1], maxpooling_size=[2, 2, 2, 2, 2],
                  maxpooling_stride=[2, 2, 2, 2, 2], maxpooling_pad='SAME', class_num=10,
                  batch_size=50, epochs=500, lr=0.001, lr_end = 0.0001, 
                  Random_noise=True, gaussain_noise_std=0.25, Adversarial_noise=False, epsilon=0,
                  adversary_target_cls=3, Targeted=False,
                  Training=False, continue_training=False, saved_model_epochs=500):
    PATH = './saved_models/cnn_epoch_{}/'.format(epochs)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0 
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    one_hot_y_train = tf.one_hot(np.squeeze(y_train).astype(np.float32), depth=class_num)
    one_hot_y_test = tf.one_hot(np.squeeze(y_test).astype(np.float32), depth=class_num)
    tr_dataset = tf.data.Dataset.from_tensor_slices((x_train, one_hot_y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, one_hot_y_test)).batch(batch_size)

    cnn_model = Deterministic_CNN(kernel_size=kernels_size, num_kernel=num_kernels, pooling_size=maxpooling_size,
                                 pooling_stride=maxpooling_stride, pooling_pad=maxpooling_pad, units=class_num,
                                 name='Dcnn')
    num_train_steps = epochs * int(x_train.shape[0] / batch_size)
    #    step = min(step, decay_steps)
    #    ((initial_learning_rate - end_learning_rate) * (1 - step / decay_steps) ^ (power) ) + end_learning_rate
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr, decay_steps=num_train_steps,  end_learning_rate=lr_end, power=2.)    
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn)  # , clipnorm=1.0)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    @tf.function  # Make it fast.
    def train_on_batch(x, y):
        with tf.GradientTape() as tape:
            cnn_model.trainable = True  
            out = cnn_model(x, training=True)                                    
            loss = loss_fn(y, out)                      
            gradients = tape.gradient(loss, cnn_model.trainable_weights)          

          #  gradients = [(tf.where(tf.math.is_nan(grad), tf.constant(1.0e-5, shape=grad.shape), grad)) for grad in gradients]
          #  gradients = [(tf.where(tf.math.is_inf(grad), tf.constant(1.0e-5, shape=grad.shape), grad)) for grad in gradients]
        optimizer.apply_gradients(zip(gradients, cnn_model.trainable_weights))       
        return loss, out
        
    @tf.function
    def validation_on_batch(x, y):       
        cnn_model.trainable = False   
        out = cnn_model(x, training=False)        
        total_vloss = loss_fn(y, out)             
        return total_vloss, out 
    @tf.function
    def test_on_batch(x, y):  
        cnn_model.trainable = False                    
        out = cnn_model(x, training=False)          
        return out
    @tf.function    
    def create_adversarial_pattern(input_image, input_label):
          with tf.GradientTape() as tape:
            tape.watch(input_image)
            cnn_model.trainable = False 
            prediction = cnn_model(input_image)                                     
            loss = loss_fn(input_label, prediction)           
          # Get the gradients of the loss w.r.t to the input image.
          gradient = tape.gradient(loss, input_image)
          # Get the sign of the gradients to create the perturbation
          signed_grad = tf.sign(gradient)
          return signed_grad  
    if Training: 
       # wandb.init(entity = "dimah", project="DCNN_Cifar10_11layers_epochs_{}_lr_{}_latest".format(epochs, lr)) 
        if continue_training:
            saved_model_path = './saved_models/cnn_epoch_{}/'.format(saved_model_epochs)
            cnn_model.load_weights(saved_model_path + 'Deterministic_cnn_model')
        
        train_acc = np.zeros(epochs)
        valid_acc = np.zeros(epochs)        
        train_err = np.zeros(epochs)
        valid_err = np.zeros(epochs)                      
        start = timeit.default_timer()
        for epoch in range(epochs):
            print('Epoch: ', epoch + 1, '/', epochs)            
            tr_no_steps = 0
            va_no_steps = 0
            # -------------Training--------------------
            acc_training = np.zeros(int(x_train.shape[0] / (batch_size)))
            err_training = np.zeros(int(x_train.shape[0] / (batch_size)))            
            for step, (x, y) in enumerate(tr_dataset):
                update_progress(step / int(x_train.shape[0] / (batch_size)))
                loss, out = train_on_batch(x, y)             
                err_training[tr_no_steps] = loss.numpy()              
                corr = tf.equal(tf.math.argmax(out, axis=-1), tf.math.argmax(y, axis=-1))
                accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))                  
                acc_training[tr_no_steps] = accuracy.numpy()                                               
                tr_no_steps += 1                
              
            train_acc[epoch] = np.mean(np.amax(acc_training))
            train_err[epoch] = np.mean(np.amin(err_training))
            print('Training Acc  ', train_acc[epoch])
            print('Training error', train_err[epoch])                                       
            # ---------------Validation----------------------  
            acc_validation = np.zeros(int(x_test.shape[0] / (batch_size)))
            err_validation = np.zeros(int(x_test.shape[0] / (batch_size)))                     
            for step, (x, y) in enumerate(val_dataset):
                update_progress(step / int(x_test.shape[0] / (batch_size)))
                total_vloss, out = validation_on_batch(x, y)                   
                err_validation[va_no_steps] = total_vloss.numpy()              
                corr = tf.equal(tf.math.argmax(out, axis=-1), tf.math.argmax(y, axis=-1))
                va_accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))                 
                acc_validation[va_no_steps] = va_accuracy.numpy()                
                va_no_steps += 1               
            
            valid_acc[epoch] = np.mean(np.amax(acc_validation))
            valid_err[epoch] = np.mean(np.amin(err_validation))           
            stop = timeit.default_timer() 
            cnn_model.save_weights(PATH + 'Deterministic_cnn_model')                   
##            wandb.log({"Training Loss":  train_err[epoch],                        
##                       "Training Accuracy": train_acc[epoch],                                             
##                        "Validation Loss": valid_err[epoch],                        
##                        "Validation Accuracy": valid_acc[epoch],                       
##                        'epoch': epoch
##                       })             
            print('Total Training Time: ', stop - start)
            print(' Training Acc   ', train_acc[epoch])            
            print(' Validation Acc ', valid_acc[epoch])            
            print('------------------------------------')
            print('Training error   ', train_err[epoch])            
            print('Validation error', valid_err[epoch])                    
            # -----------------End Training--------------------------
        cnn_model.save_weights(PATH + 'Deterministic_cnn_model')
        if (epochs > 1):
            xnew = np.linspace(0, epochs-1, 20) 
            train_spl = make_interp_spline(np.array(range(0, epochs)), train_acc)
            train_acc1 = train_spl(xnew)
            valid_spl = make_interp_spline(np.array(range(0, epochs)), valid_acc)
            valid_acc1 = valid_spl(xnew)             
            fig = plt.figure(figsize=(15, 7))            
            plt.plot(xnew, train_acc1, 'b', label='Training acc')
            plt.plot(xnew, valid_acc1, 'r', label='Validation acc')
            plt.ylim(0, 1.1)
            plt.title("Deterministic CNN on CIFAR-10 Data")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='lower right')
            plt.savefig(PATH + 'CNN_on_CIFAR10_Data_acc.png')
            plt.close(fig)             
            
            train_spl = make_interp_spline(np.array(range(0, epochs)), train_err)
            train_err1 = train_spl(xnew)
            valid_spl = make_interp_spline(np.array(range(0, epochs)), valid_err)
            valid_err1 = valid_spl(xnew)  
            fig = plt.figure(figsize=(15, 7))
            plt.plot(xnew, train_err1, 'b', label='Training loss')
            plt.plot(xnew, valid_err1, 'r', label='Validation loss')            
            plt.title("Deterministic CNN on CIFAR-10 Data")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend(loc='upper right')
            plt.savefig(PATH + 'CNN_on_CIFAR10_Data_error.png')
            plt.close(fig)
        
        f1 = open(PATH + 'training_validation_acc_error.pkl', 'wb')
        pickle.dump([train_acc, valid_acc, train_err, valid_err], f1)
        f1.close()

        textfile = open(PATH + 'Related_hyperparameters.txt', 'w')
        textfile.write(' Input Dimension : ' + str(input_dim))
        textfile.write('\n No of Kernels : ' + str(num_kernels))
        textfile.write('\n Number of Classes : ' + str(class_num))
        textfile.write('\n No of epochs : ' + str(epochs))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end))
        textfile.write('\n kernels Size : ' + str(kernels_size))
        textfile.write('\n Max pooling Size : ' + str(maxpooling_size))
        textfile.write('\n Max pooling stride : ' + str(maxpooling_stride))
        textfile.write('\n batch size : ' + str(batch_size))        
        textfile.write("\n---------------------------------")
        if Training:
            textfile.write('\n Total run time in sec : ' + str(stop - start))
            if (epochs == 1):
                textfile.write("\n Training Accuracy : " + str(train_acc))                
                textfile.write("\n Validation Accuracy : " + str(valid_acc))                
                textfile.write("\n Training error : " + str(train_err))                
                textfile.write("\n Validation error : " + str(valid_err))                
            else:
                textfile.write("\n Training Accuracy : " + str(np.mean(train_acc )))                
                textfile.write("\n Validation Accuracy : " + str(np.mean(valid_acc)))                
                textfile.write("\n Training error : " + str(np.mean(train_err)))               
                textfile.write("\n Validation error : " + str(np.mean(valid_err)))                
        textfile.write("\n---------------------------------")
        textfile.write("\n---------------------------------")
        textfile.close()
    # -------------------------Testing-----------------------------
    elif (Random_noise):
        test_path = 'test_results_random_noise_{}/'.format(gaussain_noise_std)
        cnn_model.load_weights(PATH + 'Deterministic_cnn_model')
        test_no_steps = 0
       
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, n_channels])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        out_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])        
        acc_test = np.zeros(int(x_test.shape[0] / (batch_size)))
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(x_test.shape[0] / (batch_size)))
            true_x[test_no_steps, :, :, :] = x
            true_y[test_no_steps, :, :] = y
            if Random_noise:
                noise = tf.random.normal(shape=[batch_size, input_dim, input_dim, 1], mean=0.0,
                                         stddev=gaussain_noise_std, dtype=x.dtype)
                x = x + noise
            out   = test_on_batch(x, y)            
            out_[test_no_steps, :, :] = out            
            corr = tf.equal(tf.math.argmax(out, axis=-1), tf.math.argmax(y, axis=-1))
            accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
            acc_test[test_no_steps] = accuracy.numpy()            
            test_no_steps += 1
            
        test_acc = np.mean(np.amax(acc_test)  )      
        print('Test accuracy : ', test_acc)        

        pf = open(PATH + test_path + 'info.pkl', 'wb')
        pickle.dump([out_, true_x, true_y, test_acc], pf)
        pf.close()

        textfile = open(PATH + test_path + 'Related_hyperparameters.txt', 'w')
        textfile.write(' Input Dimension : ' + str(input_dim))
        textfile.write('\n No of Kernels : ' + str(num_kernels))
        textfile.write('\n Number of Classes : ' + str(class_num))
        textfile.write('\n No of epochs : ' + str(epochs))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end))
        textfile.write('\n kernels Size : ' + str(kernels_size))
        textfile.write('\n Max pooling Size : ' + str(maxpooling_size))
        textfile.write('\n Max pooling stride : ' + str(maxpooling_stride))
        textfile.write('\n batch size : ' + str(batch_size))        
        textfile.write("\n---------------------------------")
        textfile.write("\n Test Accuracy : " + str(test_acc))
        textfile.write("\n---------------------------------")
        if Random_noise:
            textfile.write('\n Random Noise std: ' + str(gaussain_noise_std))
        textfile.write("\n---------------------------------")
        textfile.close()
    elif (Adversarial_noise):
        if Targeted:
            test_path = 'test_results_targeted_adversarial_noise_{}/'.format(epsilon)
        else:
            test_path = 'test_results_non_targeted_adversarial_noise_{}/'.format(epsilon)
        cnn_model.load_weights(PATH + 'Deterministic_cnn_model')
        cnn_model.trainable = False        

        test_no_steps = 0        
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, n_channels])
        adv_perturbations = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, n_channels])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        out_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])        
        acc_test = np.zeros(int(x_test.shape[0] / (batch_size)))
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(x_test.shape[0] / (batch_size)))
            true_x[test_no_steps, :, :, :] = x
            true_y[test_no_steps, :, :] = y
            if Targeted:
                y_true_batch = np.zeros_like(y)
                y_true_batch[:, adversary_target_cls] = 1.0
                adv_perturbations[test_no_steps, :, :, :] = (-1)*create_adversarial_pattern(x, y_true_batch)
            else:
                adv_perturbations[test_no_steps, :, :, :] = create_adversarial_pattern(x, y)
            adv_x = x + epsilon * adv_perturbations[test_no_steps, :, :, :]
            adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)
            
            out   = test_on_batch(adv_x, y)            
            out_[test_no_steps, :, :] = out            
            corr = tf.equal(tf.math.argmax(out, axis=-1), tf.math.argmax(y, axis=-1))
            accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
            acc_test[test_no_steps]=accuracy.numpy()            
            test_no_steps += 1
        test_acc = np.mean(np.amax(acc_test)   )         
        print('Test accuracy : ', test_acc)        

        pf = open(PATH + test_path + 'info.pkl', 'wb')
        pickle.dump([out_, true_x, true_y, adv_perturbations, test_acc], pf)
        pf.close()

        textfile = open(PATH + test_path + 'Related_hyperparameters.txt', 'w')
        textfile.write(' Input Dimension : ' + str(input_dim))
        textfile.write('\n No of Kernels : ' + str(num_kernels))
        textfile.write('\n Number of Classes : ' + str(class_num))
        textfile.write('\n No of epochs : ' + str(epochs))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end))
        textfile.write('\n kernels Size : ' + str(kernels_size))
        textfile.write('\n Max pooling Size : ' + str(maxpooling_size))
        textfile.write('\n Max pooling stride : ' + str(maxpooling_stride))
        textfile.write('\n batch size : ' + str(batch_size))        
        textfile.write("\n---------------------------------")
        textfile.write("\n Test Accuracy : " + str(test_acc))
        textfile.write("\n---------------------------------")
        if Adversarial_noise:
            if Targeted:
                textfile.write('\n Adversarial attack: TARGETED')
                textfile.write('\n The targeted attack class: ' + str(adversary_target_cls))
            else:
                textfile.write('\n Adversarial attack: Non-TARGETED')
            textfile.write('\n Adversarial Noise epsilon: ' + str(epsilon))
        textfile.write("\n---------------------------------")
        textfile.close()
if __name__ == '__main__':
    main_function()
