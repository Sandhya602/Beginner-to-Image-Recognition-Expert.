import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_mnist(normalize=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., None]
    x_test = x_test[..., None]
    if normalize:
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
    num_classes = 10
    input_shape = (28, 28, 1)
    return (x_train, y_train), (x_test, y_test), input_shape, num_classes

def get_augmented_flow(x, y, batch_size=64):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    datagen.fit(x)
    y_cat = tf.keras.utils.to_categorical(y, 10)
    return datagen.flow(x, y_cat, batch_size=batch_size)

def get_directory_generators(train_dir, val_dir, img_size=(224,224), batch_size=32, class_mode="categorical"):
    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    val_aug = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=class_mode
    )
    val_gen = val_aug.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False
    )
    input_shape = (*img_size, 3)
    num_classes = train_gen.num_classes
    return train_gen, val_gen, input_shape, num_classes
  
