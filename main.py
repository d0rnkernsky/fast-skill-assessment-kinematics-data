from mhadatareader import MhaDataReader
from classes import ParticipantsData, Scan, ProficiencyLabel, FoldSplit
import utils as ut
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import random
from tensorflow.keras.utils import plot_model
from sklearn.metrics import recall_score, confusion_matrix
import seaborn as sn
import pandas as pd

import os

DIR_NAME = './data_bckp'
MODEL_NAME = 'best_model.tf'
LOG_DIR = './logs'


def save_model(model, fold):
    model.save(f'./model_{fold}.tf')


def build_model(input_shape, num_classes, filters, kernel_size, dropout_rate, regularizer):
    input_layer = keras.layers.Input(shape=input_shape)

    conv1 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same",
                                kernel_regularizer=regularizer,
                                activation='relu')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Dropout(dropout_rate)(conv1)

    conv2 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                padding="same", kernel_regularizer=regularizer,
                                activation='relu')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Dropout(dropout_rate)(conv2)

    conv3 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                padding="same", kernel_regularizer=regularizer,
                                activation='relu')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Dropout(dropout_rate)(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)
    output_layer = keras.layers.Dense(num_classes, activation='softmax')(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def load_model(num, reg):
    build = True

    m_name_tuned = f'./model_tuned_{num}_{reg}.tf'
    if os.path.exists(m_name_tuned):
        print(f'loading prev tuned {m_name_tuned}')
        return keras.models.load_model(m_name_tuned), not build

    m_name_untuned = f'./model_{num}.tf'
    print(f'loading non-tuned {m_name_untuned}')
    return keras.models.load_model(m_name_untuned), build


def save_model_tune(model, num, reg):
    m_name_tuned = f'./model_tuned_{num}_{reg}.tf'

    print(f'saving {m_name_tuned}')
    model.save(m_name_tuned)


def build_model_funetune(base_model, input_shape, num_classes, filters,
                         kernel_size, dropout_rate, regularizer):
    assert base_model is not None
    model = keras.Sequential()

    # freeze the base model
    base_model.trainable = False
    # add all layers except the last two
    for layer in base_model.layers[:-2]:
        model.add(layer)

    model.add(keras.layers.Conv1D(filters=filters,
                                  kernel_size=kernel_size, padding="same",
                                  kernel_regularizer=regularizer,
                                  activation='relu',
                                  name=f'Conv1D_{str(len(model.layers) + 1)}'))
    model.add(keras.layers.BatchNormalization(
        name=f'BatchNormalization_{str(len(model.layers) + 1)}'
    ))
    model.add(keras.layers.Dropout(dropout_rate,
                                   name=f'Dropout_{str(len(model.layers) + 1)}'
                                   ))

    model.add(keras.layers.GlobalAveragePooling1D(name=f'GlobalAveragePooling1D_{str(len(model.layers) + 1)}'))
    model.add(keras.layers.Dense(num_classes, activation='softmax',
                                 name=f'Dense_{str(len(model.layers) + 1)}'))

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])

    return model


if __name__ == '__main__':
    novices_all, intermed_all, experts_all = ut.load_data(DIR_NAME)
    prepared = ut.prepare_data(novices_all, intermed_all, experts_all, incl_extra=False)
    slice_window = 70
    print(slice_window)

    folds_stats = []
    models_train_hist = dict()

    # hyper-parameters #
    kernel_size = 5
    filters = 64
    epochs = 300
    batch_size = 32
    dropout_rate = 0.5
    learning_rate = 0.0001

    CALLBACKS = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=0.000001),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            verbose=1,
        )
    ]
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
    )
    iterations = list(it.permutations([i for i in range(len(prepared[ut.Scan.ALL]))]))
    regularizer = keras.regularizers.l1_l2(0.05)

    novices_all, intermed_all, experts_all = prepared[ut.Scan.ALL]
    folds_all = ut.form_folds(novices_all, intermed_all, experts_all, repeat_experts=False)

    model_1 = None
    train, valid, test = iterations[1]
    train, valid, test = folds_all[train], folds_all[valid], folds_all[test]
    train, valid, test = ut.prepare_folds(train, valid, test, slice_window)

    x_train, y_train = train
    x_val, y_val = valid
    x_test, y_test = test

    model = build_model(
        x_train.shape[1:],
        len(ProficiencyLabel),
        kernel_size=kernel_size,
        filters=filters,
        dropout_rate=dropout_rate,
        regularizer=regularizer
    )
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )
    # plot_model(model_1, to_file='model.png')
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=CALLBACKS,
        validation_data=(x_val, y_val),
        verbose=1,
    )

    models_train_hist[0] = history.history

    #test_loss, test_acc = model.evaluate(x_test, y_test)
    #print('Test accuracy', test_acc)
    #print('Test loss', test_loss)

    # save_model(model, 0)

    y_pred = model.predict(x_test)

    y_test_cat = keras.utils.to_categorical(y_test)
    y_test_cat = y_test_cat.argmax(axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    matrix = confusion_matrix(y_test_cat, y_pred, labels=[2, 1, 0])
    #true (rows), predicted (columns)

    labels = ["Novice", "Intermediate", "Expert"]
    df_cm = pd.DataFrame(matrix, range(3), range(3))
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},
               xticklabels=labels,
               yticklabels=labels,
               cbar=False, cmap="gray") # font size

    plt.show()

"""
    folds_stats.append((test_loss, test_acc))

    ##### FINE TUNING
    fine_models_train_hist = dict()
    fine_folds_stats = dict()

    for i in range(6):
        fine_models_train_hist[i] = dict()
        fine_folds_stats[i] = dict()
        for _, reg in enumerate([Scan.LUQ, Scan.RUQ, Scan.PERICARD, Scan.PELVIC]):
            fine_models_train_hist[i][reg] = dict()
            fine_folds_stats[i][reg] = []

    # hyper-parameters #
    fine_epochs = 50
    fine_batch_size = 32
    fine_kernel_size = 5
    fine_filters = 64
    fine_dropout_rate = 0.5
    fine_learning_rate = 0.00001
    fine_regularizer = keras.regularizers.l1_l2(0.05)
    fine_CALLBACKS = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=20, min_lr=0.000001
        ),
    ]

    i = 0
    for _, reg in enumerate([Scan.LUQ, Scan.RUQ, Scan.PERICARD, Scan.PELVIC]):
        novices_reg, intermed_reg, experts_reg = prepared[reg]
        folds_reg = ut.form_folds(novices_reg, intermed_reg, experts_reg)

        train, valid, test = iterations[0]
        train, valid, test = folds_reg[train], folds_reg[valid], folds_reg[test]
        train, valid, test = ut.prepare_folds(train, valid, test, slice_window)

        x_train, y_train = train
        x_val, y_val = valid
        x_test, y_test = test

        print(f'Running fold on {reg} {i + 1}')
        print(f'Loading model {i} for region {reg}')
        model, build = load_model(i, reg.name)
        if build:
            print(f'building model {reg} {i}')
            model = build_model_funetune(
                model,
                x_train.shape[1:],
                len(ProficiencyLabel),
                kernel_size=fine_kernel_size,
                filters=fine_filters,
                dropout_rate=fine_dropout_rate,
                regularizer=fine_regularizer,
            )

        history = model.fit(
            x_train,
            y_train,
            batch_size=fine_batch_size,
            epochs=fine_epochs,
            callbacks=fine_CALLBACKS,
            validation_data=(x_val, y_val),
            verbose=1,
        )
        fine_models_train_hist[i][reg] = history.history

        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(f'Test accuracy {reg}', test_acc)
        print(f'Test loss {reg}', test_loss)

        # save_model_tune(model, i, reg)

        fine_folds_stats[i][reg].append((test_loss, test_acc))
"""