import os
import numpy as np
import math

def rotation_matrix(axis, theta):
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def data_preprocessing(values):
    # Fix the center joint position
    center_joint = values[:, 1:2, :].copy()
    values -= center_joint
    # # Parallel the bone between hip(jpt 0) and spine(jpt 1) to the z axis
    # joint_bottom = values[0, 0]
    # joint_top = values[0, 1]
    # axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
    # angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
    # matrix_z = rotation_matrix(axis, angle)
    # temp_values = np.zeros_like(values)
    # for i in range(np.shape(values)[0]):
    #     for j in range(np.shape(values)[1]):
    #         temp_values[i][j] = np.dot(matrix_z, values[i][j])
    # values = temp_values
    # # Parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) to the x axis
    # joint_rshoulder = values[0, 8]
    # joint_lshoulder = values[0, 4]
    # axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
    # angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
    # matrix_x = rotation_matrix(axis, angle)
    # temp_values = np.zeros_like(values)
    # for i in range(np.shape(values)[0]):
    #     for j in range(np.shape(values)[1]):
    #         temp_values[i][j] = np.dot(matrix_x, values[i][j])
    # values = temp_values
    return values

dir = 'raw_npy'
# dir = 'raw_npy_120'
train_ntus_subject = []
train_ntus_label_subject = []
train_ntus_len_subject = []
test_ntus_subject = []
test_ntus_label_subject = []
test_ntus_len_subject = []
train_ntus_view = []
train_ntus_label_view = []
train_ntus_len_view = []
test_ntus_view = []
test_ntus_label_view = []
test_ntus_len_view = []
train_ntus_setup = []
train_ntus_label_setup = []
train_ntus_len_setup = []
test_ntus_setup = []
test_ntus_label_setup = []
test_ntus_len_setup = []
train_subject = np.array([1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103])
train_view = np.array([2, 3])
train_setup = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31])
one_actor_seq = []
different_actor_seq = []

max_seq_len = 300
count = 0

for filename in os.listdir(dir):
    if filename.endswith('.skeleton.npy'):
        file_combiner = np.zeros((max_seq_len, 50, 3))
        data = np.load(dir+'/'+filename, allow_pickle=True).item()
        for keys, values in data.items():
            if 'nbodys' in keys:
                if all(element == 1 for element in values):
                    one_actor_seq.append(filename)
            if 'skel_body0' in keys:
                body = data_preprocessing(values)
        combine_body = True
        for keys, values in data.items():
            if 'skel_body1' in keys:
                values = data_preprocessing(values)
                body = np.concatenate((body, values), axis=1)
            elif filename in one_actor_seq and combine_body:
                body = np.concatenate((body, np.zeros_like(body)), axis=1)
                combine_body = False
        file_combiner[:len(body)] += body
        file_combiner[len(body):] += body[-1]
        label = int(filename[filename.find('A') + 1:filename.find('A') + 4])
        subject = int(filename[filename.find('P') + 1:filename.find('P') + 4])
        view = int(filename[filename.find('C') + 1:filename.find('C') + 4])
        setup = int(filename[filename.find('S') + 1:filename.find('S') + 4])
        if subject not in train_subject:
            test_ntus_subject.append(file_combiner)
            test_ntus_label_subject.append(label)
            test_ntus_len_subject.append(len(body))
        else:
            train_ntus_subject.append(file_combiner)
            train_ntus_label_subject.append(label)
            train_ntus_len_subject.append(len(body))
        if view not in train_view:
            test_ntus_view.append(file_combiner)
            test_ntus_label_view.append(label)
            test_ntus_len_view.append(len(body))
        else:
            train_ntus_view.append(file_combiner)
            train_ntus_label_view.append(label)
            train_ntus_len_view.append(len(body))
        if setup not in train_setup:
            test_ntus_setup.append(file_combiner)
            test_ntus_label_setup.append(label)
            test_ntus_len_setup.append(len(body))
        else:
            train_ntus_setup.append(file_combiner)
            train_ntus_label_setup.append(label)
            train_ntus_len_setup.append(len(body))
    count += 1
    if count % 100 == 0:
        print("checking npy ", count)

print("train_ntus_subject shape", np.shape(train_ntus_subject))
print("test_ntus_subject shape", np.shape(test_ntus_subject))
np.save('cross-subject/train_ntus.npy', train_ntus_subject)
np.save('cross-subject/train_ntus_label.npy', train_ntus_label_subject)
np.save('cross-subject/train_ntus_len.npy', train_ntus_len_subject)
np.save('cross-subject/test_ntus.npy', test_ntus_subject)
np.save('cross-subject/test_ntus_label.npy', test_ntus_label_subject)
np.save('cross-subject/test_ntus_len.npy', test_ntus_len_subject)

print("train_ntus_view shape", np.shape(train_ntus_view))
print("test_ntus_view shape", np.shape(test_ntus_view))
np.save('cross-view/train_ntus.npy', train_ntus_view)
np.save('cross-view/train_ntus_label.npy', train_ntus_label_view)
np.save('cross-view/train_ntus_len.npy', train_ntus_len_view)
np.save('cross-view/test_ntus.npy', test_ntus_view)
np.save('cross-view/test_ntus_label.npy', test_ntus_label_view)
np.save('cross-view/test_ntus_len.npy', test_ntus_len_view)

print("train_ntus_setup shape", np.shape(train_ntus_setup))
print("test_ntus_setup shape", np.shape(test_ntus_setup))
np.save('cross-setup/train_ntus.npy', train_ntus_setup)
np.save('cross-setup/train_ntus_label.npy', train_ntus_label_setup)
np.save('cross-setup/train_ntus_len.npy', train_ntus_len_setup)
np.save('cross-setup/test_ntus.npy', test_ntus_setup)
np.save('cross-setup/test_ntus_label.npy', test_ntus_label_setup)
np.save('cross-setup/test_ntus_len.npy', test_ntus_len_setup)

