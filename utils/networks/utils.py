from tensorflow.python.keras.layers import Concatenate, Lambda
from tensorflow.python.keras.models import Model
import numpy as np


def make_parallel(model, gpu_count):
    import tensorflow as tf

    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = Concatenate(axis=0)(outputs_all[0])
        #for outputs in outputs_all:
        #    merged.append(merge(outputs, mode='concat', concat_axis=0))
            
    return Model(inputs=model.inputs, outputs=merged)

def predict_with_overlap(model, image, input_dimensions=(150,150,150), overlap=50):
    prediction=np.zeros(image.shape,dtype='float32')
    zdim,xdim,ydim=input_dimensions
    
    i=0
    total=len(range(0, image.shape[0], zdim-overlap))*len(range(0, image.shape[1], xdim-overlap))*len(range(0, image.shape[2], ydim-overlap))
    
    for z in range(0, image.shape[0], zdim-overlap):
        for x in range(0, image.shape[1], xdim-overlap):
            for y in range(0, image.shape[2], ydim-overlap):
                prediction_part=model.predict(np.expand_dims(image[z:z+zdim,x:x+xdim,y:y+ydim],axis=0))[0]
                prediction[z:z+zdim,x:x+xdim,y:y+ydim]=np.maximum(prediction[z:z+zdim,x:x+xdim,y:y+ydim],prediction_part)
                i+=1
                printProgressBar(i + 1, total, prefix = 'Progress:', suffix = 'Complete', length = 50)
    return prediction

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()