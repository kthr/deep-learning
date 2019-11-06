from tensorflow.python.keras.layers import Input, Dropout, BatchNormalization, Conv2D, Conv3D, Concatenate
from tensorflow.python.keras.models import Model

def MSD_2D(width, depth, input_shape=(None,None,1), activation='sigmoid', kernel_size=(3,3), max_dilations=10, n_output_channels=1, drop_out=0.0, dropout_at_prediction=False, batch_norm=False):
    inp = Input(shape=input_shape)
    inputs = [inp]
    
    receptive_field=1
    for i in range(depth):
        for j in range(width): 
            s_ij = ((i*width + j) % max_dilations) + 1
            c = _convolution_2D(n_filters=1, dilation=s_ij, inputs=inputs, kernel_size=kernel_size, drop_out=drop_out, dropout_at_prediction=dropout_at_prediction, batch_norm=batch_norm, name='layer_{}_dilation_{}'.format((i*width + j), s_ij))
            inputs.append(c)
            receptive_field=_receptive_field(receptive_field,max(kernel_size),s_ij)
        
    c = Concatenate()(inputs)
    o = Conv2D(filters=n_output_channels, kernel_size=(1,1), padding='same', activation=activation)(c)
    
    print("Maximum theoretical receptive field of {} pixels".format(receptive_field))
    model = Model(inputs=inp, outputs=[o])
    return model

def MSD_3D(width, depth, input_shape=(None,None,None,1), activation='sigmoid', kernel_size=(3,3,3), max_dilations=10, n_output_channels=1, drop_out=0.0, dropout_at_prediction=False, batch_norm=False):
    inp = Input(shape=input_shape)
    inputs = [inp]
    
    receptive_field=1
    for i in range(depth):
        for j in range(width): 
            s_ij = ((i*width + j) % max_dilations) + 1
            c = _convolution_3D(n_filters=1, dilation=s_ij, inputs=inputs, kernel_size=kernel_size, drop_out=drop_out, dropout_at_prediction=dropout_at_prediction, batch_norm=batch_norm, name='layer_{}_dilation_{}'.format((i*width + j), s_ij))
            inputs.append(c)
            receptive_field=_receptive_field(receptive_field,max(kernel_size),s_ij)
        
    c = Concatenate()(inputs)
    o = Conv3D(filters=n_output_channels, kernel_size=(1,1,1), padding='same', activation=activation)(c)
    
    print("Maximum theoretical receptive field of {} pixels".format(receptive_field))
    model = Model(inputs=inp, outputs=[o])        
    return model

def _convolution_2D(n_filters, dilation, inputs, kernel_size, drop_out, dropout_at_prediction, batch_norm, name=None):
        if len(inputs) > 1:
            i = Concatenate()(inputs)
        else:
            i = inputs[0]
        c = Conv2D(filters=n_filters, dilation_rate=(dilation, dilation),
                                        kernel_size=kernel_size, strides=(1,1), padding='same', activation='relu', use_bias=True, name=name)(i)
        if batch_norm:
            c = BatchNormalization()(c)
            
        if drop_out:
            if dropout_at_prediction:
                c = Dropout(rate=drop_out)(c, training=True)
            else:
                c = Dropout(rate=drop_out)(c)
            
        return c
    
def _convolution_3D(n_filters, dilation, inputs, kernel_size, drop_out, dropout_at_prediction, batch_norm, name=None):
        if len(inputs) > 1:
            i = Concatenate()(inputs)
        else:
            i = inputs[0]
        c = Conv3D(filters=n_filters, dilation_rate=(dilation, dilation, dilation),
                                        kernel_size=kernel_size, strides=(1,1,1), padding='same', 
                                        activation='relu', use_bias=True, name=name)(i)
        if batch_norm:
            c = BatchNormalization()(c)
            
        if drop_out:
            if dropout_at_prediction:
                c = Dropout(rate=drop_out)(c, training=True)
            else:
                c = Dropout(rate=drop_out)(c)
            
        return c
    
def _receptive_field(previous, kernel_size, dilation):
    return ((kernel_size-1)*dilation+1)+(previous-1)