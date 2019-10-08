from cv_utils.processors import ComputeHog, SobelX, SobelY, Resize
from cv_utils.utils import simple_dataloader
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, InputLayer
from tensorflow.keras.utils import to_categorical


def build_mlp(features):
    """build a mlp model

    Arguments:
        features {list} -- int numbers for each layer e.g.[1000, 2000, 1000, 8]
    
    Returns:
        tf.keras.Model -- mlp model
    """
    layers = []
    for fi, feature_num in enumerate(features):
        if fi == 0:
            layer = [InputLayer(features[fi])]
        elif fi + 1 == len(features):
            layer = [Dense(features[fi], activation='softmax')]
        else:
            layer = [
                Dense(features[fi], use_bias=True),
                Dropout(0.5),
                BatchNormalization(),
                Activation('sigmoid')
            ]
        layers += layer
    return Sequential(layers)


def train_mlp(train_dir, val_dir):
    processor_list = [ComputeHog(), SobelX(), SobelY(), Resize(size=(16, 16))]
    train_inputs, train_targets = simple_dataloader(
        train_dir, processor_list=processor_list)
    val_inputs, val_targets = simple_dataloader(val_dir,
                                                processor_list=processor_list)
    train_targets = to_categorical(train_targets)
    val_targets = to_categorical(val_targets)
    model = build_mlp([
        train_inputs.shape[1], 100, 1000,
        train_targets.shape[1]
    ])
    model.summary()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(x=train_inputs,
              y=train_targets,
              batch_size=64,
              epochs=100,
              validation_data=(val_inputs, val_targets),
    )


if __name__ == "__main__":
    train_dir = 'data/road_mark/train'
    val_dir = 'data/road_mark/val'
    train_mlp(train_dir, val_dir)