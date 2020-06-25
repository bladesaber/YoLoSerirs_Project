import numpy as np

# anchors = np.array([10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326])
# anchors = anchors.reshape((-1, 2))
# print(anchors)
# print(anchors[:, 0]*anchors[:, 1])
# print(anchors[0:3, :]/8)
# print(anchors[3:6, :]/16)
# print(anchors[6:9, :]/32)

anchors = np.array(
    [
        [15, 22],
        [29, 46],
        [44, 88],

        [65, 78],
        [54, 113],
        [69, 146],

        [91, 106],
        [115, 167],
        [189, 220],
    ])
print(anchors[:, 0]*anchors[:, 1])
print(anchors[0:3, :]/8)
print(anchors[3:6, :]/16)
print(anchors[6:9, :]/32)