import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import sys
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from scipy import interp
from numpy import genfromtxt

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

n_classes=4

labels_data_file = genfromtxt(str(sys.argv[1]), delimiter=',', skip_header=0)
labels_data_DNN = genfromtxt(str(sys.argv[2]), delimiter=',', skip_header=0)

size_lab=file_len(str(sys.argv[1]))
size_D=file_len(str(sys.argv[2]))

num_rows = labels_data_file.shape[0] # Number of data samples
num_cols = labels_data_file.shape[1] # Length of Data Vector

num_rowsD = labels_data_DNN.shape[0] # Number of data samples
num_colsD = labels_data_DNN.shape[1] # Length of Data Vector

label_data_size=num_cols*num_rows
label_data_sizeD=num_colsD*num_rowsD

label_res = np.arange(label_data_size)
label_res = label_res.reshape(num_rows, num_cols) # 2D Matrix of data points
label_res = label_res.astype('int32')

label_resD = np.arange(label_data_sizeD)
label_resD = label_resD.reshape(num_rowsD, num_colsD) # 2D Matrix of data points
label_resD = label_resD.astype('int32')

for i in range(labels_data_file.shape[0]):
    for j in range(num_cols):
        label_res[i][j] = labels_data_file[i][j]-1

for i in range(labels_data_DNN.shape[0]):
    for j in range(num_colsD):
        label_resD[i][j] = labels_data_DNN[i][j]


fpr = dict()
tpr = dict()
roc_auc = dict()
fpr1 = dict()
tpr1 = dict()
roc_auc1 = dict()
fpr2 = dict()
tpr2 = dict()
roc_auc2 = dict()
fpr3 = dict()
tpr3 = dict()
roc_auc3 = dict()
fpr4 = dict()
tpr4 = dict()
roc_auc4 = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(label_res[:, 0], label_res[:, 4]*np.random.rand(size_lab,),i)
    roc_auc[i] = auc(fpr[i], tpr[i])
    fpr1[i], tpr1[i], _ = roc_curve(label_res[:, 0], label_res[:, 1]*np.random.rand(size_lab,),i)
    roc_auc1[i] = auc(fpr1[i], tpr1[i])
    fpr2[i], tpr2[i], _ = roc_curve(label_res[:, 0], label_res[:, 2]*np.random.rand(size_lab,),i)
    roc_auc2[i] = auc(fpr2[i], tpr2[i])
    fpr3[i], tpr3[i], _ = roc_curve(label_res[:, 0], label_res[:, 3]*np.random.rand(size_lab,),i)
    roc_auc3[i] = auc(fpr3[i], tpr3[i])
    fpr4[i], tpr4[i], _ = roc_curve(label_resD[:, 0], label_resD[:, 1]*np.random.rand(size_D,),i)
    roc_auc4[i] = auc(fpr4[i], tpr4[i])
    print(tpr[i])
fpr[0]=fpr[0]-1
fpr1[0]=fpr1[0]-1
fpr2[0]=fpr2[0]-1
fpr3[0]=fpr3[0]-1
fpr4[0]=fpr4[0]-1
all_fpr = np.unique(np.concatenate([fpr[i] for i in xrange(int(sys.argv[3]),n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in xrange(int(sys.argv[3]),n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes-int(sys.argv[3])

roc_auc_macro = auc(all_fpr, mean_tpr)

all_fpr1 = np.unique(np.concatenate([fpr1[i] for i in xrange(int(sys.argv[3]),n_classes)]))

mean_tpr1 = np.zeros_like(all_fpr1)
for i in xrange(int(sys.argv[3]),n_classes):
    mean_tpr1 += interp(all_fpr1, fpr1[i], tpr1[i])

mean_tpr1 /= n_classes-int(sys.argv[3])

roc_auc_macro1 = auc(all_fpr1, mean_tpr1)

all_fpr2 = np.unique(np.concatenate([fpr2[i] for i in xrange(int(sys.argv[3]),n_classes)]))

mean_tpr2 = np.zeros_like(all_fpr2)
for i in xrange(int(sys.argv[3]),n_classes):
    mean_tpr2 += interp(all_fpr2, fpr2[i], tpr2[i])

mean_tpr2 /= n_classes-int(sys.argv[3])

roc_auc_macro2 = auc(all_fpr2, mean_tpr2)

all_fpr3 = np.unique(np.concatenate([fpr3[i] for i in xrange(int(sys.argv[3]),n_classes)]))

mean_tpr3 = np.zeros_like(all_fpr3)
for i in xrange(int(sys.argv[3]),n_classes):
    mean_tpr3 += interp(all_fpr3, fpr3[i], tpr3[i])

mean_tpr3 /= n_classes-int(sys.argv[3])

roc_auc_macro3 = auc(all_fpr3, mean_tpr3)

all_fpr4 = np.unique(np.concatenate([fpr4[i] for i in xrange(int(sys.argv[3]),n_classes)]))

mean_tpr4 = np.zeros_like(all_fpr4)
for i in xrange(int(sys.argv[3]),n_classes):
    mean_tpr4 += interp(all_fpr4, fpr4[i], tpr4[i])

mean_tpr4 /= n_classes-int(sys.argv[3])

roc_auc_macro4 = auc(all_fpr4, mean_tpr4)

plt.figure()
lw = 2
plt.title('macro ROCs')
plt.plot(all_fpr,mean_tpr,
         label='macro DBN 10 (AUC = %0.4f)'
          % roc_auc_macro,
         color='darkorange', linestyle='-', linewidth=2)
plt.plot(all_fpr1,mean_tpr1,
         label='macro BPN 10 (AUC = %0.4f)'
         % roc_auc_macro1,
         color='black', linestyle='-', linewidth=2)
plt.plot(all_fpr2,mean_tpr2,
         label='macro LMNC 10 (AUC = %0.4f)'
         % roc_auc_macro2,
         color='green', linestyle='-', linewidth=2)
plt.plot(all_fpr3,mean_tpr3,
         label='macro SVM R=0.1 (AUC = %0.4f)'
         % roc_auc_macro3,
         color='red', linestyle='-', linewidth=2)
plt.plot(all_fpr4,mean_tpr4,
         label='macro DNN 10 (AUC = %0.4f)'
         % roc_auc_macro4,
         color='navy', linestyle='-', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.figure()
lw = 2
plt.title('Central Apnea #3 class ROCs')
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='DBN 10 (AUC = %0.4f)' % roc_auc[2])
plt.plot(fpr1[2], tpr1[2], color='black',
         lw=lw, label='BPN 10  (AUC = %0.4f)' % roc_auc1[2])
plt.plot(fpr2[2], tpr2[2], color='green',
         lw=lw, label='LMNC 10  (AUC = %0.4f)' % roc_auc2[2])
plt.plot(fpr3[2], tpr3[2], color='red',
         lw=lw, label='SVM R=0.1  (AUC = %0.4f)' % roc_auc3[2])
plt.plot(fpr4[2], tpr4[2], color='navy',
         lw=lw, label='DNN 10  (AUC = %0.4f)' % roc_auc4[2])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.figure()
lw = 2
plt.title('Breath Hold #4 class ROCs')
plt.plot(fpr[3], tpr[3], color='darkorange',
         lw=lw, label='DBN 10 (AUC = %0.4f)' % roc_auc[3])
plt.plot(fpr1[3], tpr1[3], color='black',
         lw=lw, label='BPN 10  (AUC = %0.4f)' % roc_auc1[3])
plt.plot(fpr2[3], tpr2[3], color='green',
         lw=lw, label='LMNC 10  (AUC = %0.4f)' % roc_auc2[3])
plt.plot(fpr3[3], tpr3[3], color='red',
         lw=lw, label='SVM R=0.1  (AUC = %0.4f)' % roc_auc3[3])
plt.plot(fpr4[3], tpr4[3], color='navy',
         lw=lw, label='DNN 10  (AUC = %0.4f)' % roc_auc4[3])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.show()
