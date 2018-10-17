# import
import os
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import imageio
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_data():

    X, y = [], []
    for filename in os.listdir('contrast_images'):

        # split filename
        f = filename.split('_')
        fperson = f[0].split('-')[1][:-2]
        ttype = f[3][:3]
        axis = f[-1].split('-')[1]

        # print(fperson, ttype, axis)
        if ttype == 'dif' and axis == 'ortho.png':
            image = imageio.imread(os.path.join('contrast_images', filename))
            if fperson == 'control':
                label, is_collect = 1, True
            elif fperson == 'mdd':
                label, is_collect = 2, True
            else:
                label, is_collect = 3, False

            # add to X, y
            if is_collect:
                X.append(image)
                y.append(label)

    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


X, y = get_data()

unique, counts = np.unique(y, return_counts=True)
d = dict(zip(unique, counts))
n = d[min(d, key=d.get)]
print(d)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
y_trainHot = to_categorical(y_train, num_classes=3)
y_testHot = to_categorical(y_test, num_classes=3)

# normalize to 0 to 1
X_train = np.array(X_train)
X_train = X_train/255.0

_test = np.array(X_test)
X_test = X_test/255.0

# build cnn model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(187, 525, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(3, activation='softmax'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

model.fit(X_train, y_trainHot, epochs=5)

# test prediction
preds = model.predict(X_test)

cls_pred = np.argmax(preds, axis=1)
labels = ['control', 'mdd']
print(classification_report(y_test, cls_pred, target_names=labels))
print(confusion_matrix(y_test, cls_pred))

print(str(set(y_test)))
print(str(set(cls_pred)))


# show the images
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.2, wspace=0.0)

    for i, ax in enumerate(axes.flat):
        img_shape = (187, 525, 3)
        ax.imshow(images[i].reshape(img_shape), cmap='magma')

        if cls_pred is None:
            xlabel = "True: {0}".format(labels[cls_true[i] - 1])
        else:
            xlabel = "True: {0},\nPred: {1}".format(labels[cls_true[i] - 1],
                                                    labels[cls_pred[i] - 1])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


plot_images(X_test[0:9], y_test[0:9], cls_pred[0:9])


# roc plot
fpr, tpr, _ = roc_curve(y_test, cls_pred, pos_label=2)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
