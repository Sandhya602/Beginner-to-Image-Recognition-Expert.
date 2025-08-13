from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def build_mobilenet_head(input_shape=(96,96,3), num_classes=10, train_base=False):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet', pooling='avg')
    base.trainable = train_base
    x = layers.Dense(128, activation='relu')(base.output)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base.input, outputs=out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

