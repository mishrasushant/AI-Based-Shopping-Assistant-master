import os
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt
from collections import Counter

keyword = input("Search: ")
PATH = "data/"

if keyword not in os.listdir(PATH):
    print(f"Keyword '{keyword}' not found in the dataset.")
    exit()

for category in os.listdir(PATH):
    if category == keyword:
        path = os.path.join(PATH, category)
        for image in os.listdir(path):
            img = cv2.imread(os.path.join(path, image))
            img = cv2.resize(img, (200, 200))
            cv2.imshow(category, img)
            cv2.waitKey(1000)

cv2.destroyAllWindows()

centers = [[1,1],[5,5],[8,4]]

dataset = pd.read_csv('person.csv')

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,[0]].values
name = dataset['Item_names'].tolist()

ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

gg = Counter(labels)

def find_max():
    max_value = gg[0]  # Renamed from 'max' to 'max_value'
    v = 0
    for i in range(len(gg)):
        if gg[i] > max_value:
            max_value = gg[i]
            v = i
    return v

Y = y.tolist()
L = labels.tolist()

max_label = find_max()

suggest = []
for i in range(len(labels)):
    if max_label == L[i]:
        suggest.append(Y[i])

new = []

def stripp(rr):
    local_new = []  # Use a local list instead of the global 'new'
    for i in range(len(suggest)):
        p = str(rr[i]).replace('[', '').replace(']', '')
        local_new.append(int(p))
    return local_new

new_Y = stripp(Y)
new_name = []
for i in range(len(suggest)):
    p = str(name[i]).replace('[', '').replace(']', '')
    new_name.append(p)

n_clusters_ = len(np.unique(labels))

suggest = min(10, len(new_Y))  # Ensure 'suggest' does not exceed the length of 'new_Y'
colors = 10*['r.','g.','b.','c.','k.','y.','m.']

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]],markersize = 10)

plt.scatter(cluster_centers[:,0],cluster_centers[:,1], marker = "x", s=150, linewidths = 5, zorder=10)

item_name = dict(zip(new_Y, new_name))

print("Recommendations::")
for i in range(suggest):
    print(f"Item ID- {new_Y[i]}   Item name- {new_name[i]}")

plt.show()




