# import
import csv
import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_time():

    # iterate run
    time = dict()
    for i in [1, 2, 4, 5]:

        # read file
        filename = 'event/run-' + str(i) + '.tsv'
        with open(filename, mode='r') as f:
            reader = csv.DictReader(f, delimiter='\t')

            # read row
            ton, pos, neg = [], [], []
            for row in reader:

                # check trial_type
                if row['trial_type'] != 'response':
                    ttype = row['trial_type'][:3]
                else:
                    # collect sample
                    onset = int(float(row['onset']) // 3)
                    sample = []
                    for j in [0]:
                        if onset + j < 105: sample.append(onset + j)
                    if ttype == 'ton': ton += sample
                    if ttype == 'pos': pos += sample
                    if ttype == 'neg': neg += sample
            time[i] = [ton, pos, neg]
    return time


def get_change(run, axis, depth):

    # get time
    time = get_time()

    # iterate file
    X, y = [], []
    for filename in os.listdir('dataset'):

        # split filename
        f = filename.split('_')
        fperson = f[0].split('-')[0]
        frun = int(f[1][4])

        # check run
        if frun in run:
            # read image
            nii = nib.Nifti1Image.from_filename(os.path.join('dataset', filename))
            arr = nii.get_data().T

            # get initial data
            if axis == 'x': init = arr[0, depth, :, :]
            if axis == 'y': init = arr[0, :, depth, :]
            if axis == 'z': init = arr[0, :, :, depth]

            # iterate time
            ton, pos, neg = time[frun]
            for trial_type, trial_time in [('ton', ton), ('pos', pos), ('neg', neg)]:
                for ftime in trial_time:
                    try:
                        # check axis
                        if axis == 'x': data = arr[ftime, depth, :, :]
                        if axis == 'y': data = arr[ftime, :, depth, :]
                        if axis == 'z': data = arr[ftime, :, :, depth]

                        # convert person and ttype to numeric label
                        if fperson == 'control' and trial_type == 'ton':
                            label, is_collect = 1, False
                        elif fperson == 'control' and trial_type == 'pos':
                            label, is_collect = 1, False
                        elif fperson == 'control' and trial_type == 'neg':
                            label, is_collect = 1, True
                        elif fperson == 'mdd' and trial_type == 'ton':
                            label, is_collect = 2, False
                        elif fperson == 'mdd' and trial_type == 'pos':
                            label, is_collect = 2, False
                        elif fperson == 'mdd' and trial_type == 'neg':
                            label, is_collect = 2, True
                        else:
                            label, is_collect = 3, False

                        # add to X, y
                        if is_collect:
                            change = init - data

                            X.append(change)
                            y.append(label)
                    except IndexError:
                        pass
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


# get dataset
X, y = get_change(run=[1, 2, 3, 4], axis='x', depth=20)

# balance classes
unique, counts = np.unique(y, return_counts=True)
d = dict(zip(unique, counts))
n = d[min(d, key=d.get)]

mask = np.hstack([np.random.choice(np.where(y == l)[0], n, replace=False) for l in np.unique(y)])
y = y[mask]
X = X[mask]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
y_trainHot = to_categorical(y_train, num_classes=3)
y_testHot = to_categorical(y_test, num_classes=3)

# normalize to 0 to 1
X_train = np.array(X_train)
X_train = X_train/16.0

X_test = np.array(X_test)
X_test = X_test/16.0

# build cnn model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(80, 80)))
# model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
# model.add(tf.keras.layers.Dropout(0.25))
# model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
# model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
# model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

model.fit(X_train, y_trainHot, epochs=50)

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

    # create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.2, wspace=0.0)

    for i, ax in enumerate(axes.flat):
        # plot image.
        img_shape = (80, 80)
        ax.imshow(images[i].reshape(img_shape), cmap='magma')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(labels[cls_true[i] - 1])
        else:
            xlabel = "True: {0},\nPred: {1}".format(labels[cls_true[i] - 1], labels[cls_pred[i] - 1])

        # show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


plot_images(X_test[0:9], np.argmax(y_testHot[0:9], axis=1), cls_pred[0:9])


# roc plot
fpr, tpr, _ = roc_curve(np.argmax(y_testHot, axis=1), cls_pred, pos_label=2)
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
