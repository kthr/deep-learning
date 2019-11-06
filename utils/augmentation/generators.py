from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

def xy_augmentation_generator(x, y, x_augmentation, y_augmentation, x_flow_parameters=dict(), y_flow_parameters=dict(), batch_size=32, seed=1, generator=ImageDataGenerator):
    """
    Build a image generator, where x and the corresponding y images are augmented according to the given
    augmentation parameters.

    Args:
        x (numpy.array): an array of images
        y (numpy.array): an array of images
        x_augmentation (dict): a dictionary of augmentation parameters, for the given generator
        y_augmentation (dict): a dictionary of augmentation parameters, for the given generator
        x_flow_parameters (dict): parameters to the flow function for the given generator
        y_flow_parameters (dict): parameters to the flow function for the given generator
        batch_size (int): the batch size used for training
        seed (int): a random seed for ensuring that x and y values are correctly associated

    Returns:
        (zip) returns the zipped image data generators for x and y
    """
    image_datagen = generator(**x_augmentation)
    mask_datagen = generator(**y_augmentation)

    # Provide the same seed and keyword arguments to the fit and flow methods
    if ('featurewise_center' in x_augmentation) or ('featurewise_std_normalization' in x_augmentation) or ('zca_whitening' in x_augmentation):
        image_datagen.fit(x, augment=True, seed=seed)
    if ('featurewise_center' in y_augmentation) or ('featurewise_std_normalization' in y_augmentation) or ('zca_whitening' in y_augmentation):
        mask_datagen.fit(y, augment=True, seed=seed)

    image_generator = image_datagen.flow(x, seed=seed, batch_size=batch_size, **x_flow_parameters)
    mask_generator = mask_datagen.flow(y, seed=seed, batch_size=batch_size, **y_flow_parameters)
    return zip(image_generator, mask_generator)
