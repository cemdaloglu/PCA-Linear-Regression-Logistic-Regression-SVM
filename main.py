import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy.linalg import inv
import scipy.io
from sklearn.svm import SVC
from sklearn import metrics
import time


# preprocessing
def preprocess():
    d3_picture = []
    for x in range(1, 878):
        read_picture = 'van_gogh\Vincent_van_Gogh_' + str(x) + '.jpg'
        img = mpimg.imread(read_picture)
        if img.shape == (64, 64):
            img_create = np.zeros((64, 64, 3), dtype=int)
            for y in range(0, 64):
                for z in range(0, 64):
                    rgb_matrix = [img[y, z], img[y, z], img[y, z]]
                    img_create[y, z] = img[y, z]
                    img_create[y, z, :] = rgb_matrix
            d3_picture.append(img_create)
        else:
            d3_picture.append(img)
    X = np.reshape(d3_picture, (877, 4096, 3))
    return X


# question 1.1
def question_1_1(inp):
    print("question 1.1")
    X = inp
    U_0, s_0, VT_0 = np.linalg.svd(X[:, :, 0])
    U_1, s_1, VT_1 = np.linalg.svd(X[:, :, 1])
    U_2, s_2, VT_2 = np.linalg.svd(X[:, :, 2])

    # # reconstruct
    # pc_number = 877
    # picture = np.zeros((pc_number, 4096, 3))
    # dot_1 = np.dot(U_0[:pc_number, 0:pc_number], np.diag(s_0[0:pc_number]))
    # dot_2 = np.dot(U_1[:pc_number, 0:pc_number], np.diag(s_1[0:pc_number]))
    # dot_3 = np.dot(U_2[:pc_number, 0:pc_number], np.diag(s_2[0:pc_number]))
    # picture[:, :, 0] = np.dot(dot_1, VT_0[0:pc_number,:])
    # picture[:, :, 1] = np.dot(dot_2, VT_1[0:pc_number,:])
    # picture[:, :, 2] = np.dot(dot_3, VT_2[0:pc_number,:])
    # picture = np.reshape(picture, (pc_number, 64, 64, 3))
    # picture = picture.astype('uint8')
    # plt.imshow(picture[0])
    # plt.show()

    plt.figure()
    plt.bar(range(0, 100), s_0[0:100])
    plt.xlabel("sample no")
    plt.ylabel("singular values")
    plt.title("X1")
    plt.figure()
    plt.bar(range(0, 100), s_1[0:100])
    plt.xlabel("sample no")
    plt.ylabel("singular values")
    plt.title("X2")
    plt.figure()
    plt.bar(range(0, 100), s_2[0:100])
    plt.xlabel("sample no")
    plt.ylabel("singular values")
    plt.title("X3")
    plt.show()

    summation = sum(s_0 * s_0)
    summation_1 = sum(s_1 * s_1)
    summation_2 = sum(s_2 * s_2)
    proportion_variance = np.zeros(30, dtype=float)
    for x in range(0, 10):
        proportion_variance[x] = s_0[x] * s_0[x] / summation
        proportion_variance[10 + x] = s_1[x] * s_1[x] / summation_1
        proportion_variance[20 + x] = s_2[x] * s_2[x] / summation_2
    print("proportion of variances:")
    print(proportion_variance[0:10])
    print(proportion_variance[10:20])
    print(proportion_variance[20:])


# question 1.2
def question_1_2(pimp):
    print("question 1.2")
    X_2 = pimp
    X_2 = np.reshape(X_2, (877, 64, 64, 3))
    mean = np.zeros((64, 64, 3), dtype=float)
    variance = np.zeros((64, 64, 3), dtype=float)
    summation = sum(X_2[:, :, :, :])
    mean[:, :, :] = summation / 877
    variance[:, :, :] = np.var(X_2[:, :, :, :], 0)

    gaussian_noise = np.random.normal(mean, np.sqrt(variance), (64, 64, 3))
    noisy_image = X_2 + gaussian_noise * 0.01
    noisy_image = noisy_image.astype(int)
    X_noisy = np.reshape(noisy_image, (877, 4096, 3))
    question_1_1(X_noisy)


# question 2.2
def question_2_2():
    print("question 2.2")
    with open('q2_dataset.csv', 'r') as csvfile:
        read_admission = list(csv.reader(csvfile))
    read_admission = read_admission[1:]
    np.random.shuffle(read_admission)
    read_admission = np.array(read_admission, dtype=float)
    fold = read_admission
    fold = np.array(fold, dtype=float)

    mse = []
    mae = []
    r_square = []
    mape = []
    for x in range(0, 5):
        feature_matrix_x = np.ones((400, 8), dtype=float)
        feature_matrix_test = np.ones((100, 8), dtype=float)
        output_y = np.zeros((400), dtype=float)
        feature_matrix_x = np.array(feature_matrix_x, dtype=float)
        output_y = np.array(output_y)
        feature_matrix_test[:, 1:] = fold[x * 100:(x + 1) * 100][:, 0:7]
        output_y_test = fold[x * 100:(x + 1) * 100][:, 7]
        feature_matrix_x[0:x * 100, 1:] = fold[0:x * 100, 0:7]
        feature_matrix_x[x * 100:, 1:] = fold[(x + 1) * 100:, 0:7]
        output_y[0:x * 100] = fold[0:x * 100, 7]
        output_y[x * 100:] = fold[(x + 1) * 100:, 7]
        weight_matrix_w = np.dot(inv(np.dot(feature_matrix_x.T, feature_matrix_x)),
                                 np.dot(feature_matrix_x.T, output_y))
        predictin_y = np.dot(feature_matrix_test, weight_matrix_w.T)

        mse.append(1 / 100 * sum((output_y_test - predictin_y) * (output_y_test - predictin_y)))
        mae.append(1 / 100 * sum(abs(output_y_test - predictin_y)))
        r_square.append(1 - sum((output_y_test - predictin_y) * (output_y_test - predictin_y)) / sum(
            (output_y_test - np.mean(predictin_y)) * (output_y_test - np.mean(predictin_y))))
        mape.append(1 / 100 * sum(abs((output_y_test - predictin_y) / output_y_test)))
    return mse, mae, r_square, mape


# question_2_3
def question_2_3():
    print("question 2.3")
    with open('q2_dataset.csv', 'r') as csvfile:
        read_admission = list(csv.reader(csvfile))
    read_admission = read_admission[1:]
    np.random.shuffle(read_admission)
    read_admission = np.array(read_admission, dtype=float)
    fold = read_admission
    fold = np.array(fold, dtype=float)

    L1_coeff = 0.01
    learning_rate = 0.01
    mse = []
    mae = []
    r_square = []
    mape = []
    for x in range(0, 5):
        w = np.random.randn(8) / np.sqrt(8)
        feature_matrix_x = np.ones((400, 8), dtype=float)
        feature_matrix_test = np.ones((100, 8), dtype=float)
        output_y = np.zeros(400, dtype=float)
        feature_matrix_x = np.array(feature_matrix_x, dtype=float)
        output_y = np.array(output_y)
        feature_matrix_test[:, 1:] = fold[x * 100:(x + 1) * 100][:, 0:7]
        output_y_test = fold[x * 100:(x + 1) * 100][:, 7]
        feature_matrix_x[0:x * 100, 1:] = fold[0:x * 100, 0:7]
        feature_matrix_x[x * 100:, 1:] = fold[(x + 1) * 100:, 0:7]
        output_y[0:x * 100] = fold[0:x * 100, 7]
        output_y[x * 100:] = fold[(x + 1) * 100:, 7]
        for y in range(1, 8):
            smallest = min(feature_matrix_x[:, y])
            largest = max(feature_matrix_x[:, y])
            feature_matrix_x[:, y] = (feature_matrix_x[:, y] - smallest) / (largest - smallest)
            smallest_test = min(feature_matrix_test[:, y])
            largest_test = max(feature_matrix_test[:, y])
            feature_matrix_test[:, y] = (feature_matrix_test[:, y] - smallest_test) / (largest_test - smallest_test)
        flag = True
        while flag:
            prediction_train = np.dot(feature_matrix_x, w.T)
            delta1 = prediction_train - output_y
            error1 = np.dot(delta1, delta1) / 400
            w = w - learning_rate * (2 * np.dot(feature_matrix_x.T, delta1) + L1_coeff * np.sign(w)) / 400
            prediction_train2 = np.dot(feature_matrix_x, w.T)
            delta2 = prediction_train2 - output_y
            error2 = np.dot(delta2, delta2) / 400
            if abs(error1 - error2) <= 0.00000001:
                flag = False
        weight_matrix_w = w
        prediction_y = np.dot(feature_matrix_test, weight_matrix_w.T)
        mse.append(1 / 100 * sum((output_y_test - prediction_y) * (output_y_test - prediction_y)))
        mae.append(1 / 100 * sum(abs(output_y_test - prediction_y)))
        r_square.append(1 - sum((output_y_test - prediction_y) * (output_y_test - prediction_y)) / sum(
            (output_y_test - np.mean(prediction_y)) * (output_y_test - np.mean(prediction_y))))
        mape.append(1 / 100 * sum(abs((output_y_test - prediction_y) / output_y_test)))
    return mse, mae, r_square, mape


def question_3_1_and_3_2():
    # question_3_1
    print("question 3.1")
    with open('q3_train_dataset.csv', 'r') as csvfile:
        read_train = list(csv.reader(csvfile))
    with open('q3_test_dataset.csv', 'r') as csvfile:
        read_test = list(csv.reader(csvfile))
    read_train = read_train[1:]
    read_train = np.array(read_train)
    new_train = np.ones((712, 12), dtype=float)
    new_train[:, 1:3] = read_train[:, 0:2]
    new_train[:, 5:9] = read_train[:, 3:7]
    for x in range(0, len(read_train)):
        if read_train[x, 2] == 'male':
            new_train[x, 3] = 1
            new_train[x, 4] = 0
        else:
            new_train[x, 3] = 0
            new_train[x, 4] = 1
        if read_train[x, 7] == 'S':
            new_train[x, 9] = 1
            new_train[x, 10] = 0
            new_train[x, 11] = 0
        elif read_train[x, 7] == 'C':
            new_train[x, 9] = 0
            new_train[x, 10] = 1
            new_train[x, 11] = 0
        else:
            new_train[x, 9] = 0
            new_train[x, 10] = 0
            new_train[x, 11] = 1
    read_test = read_test[1:]
    read_test = np.array(read_test)
    new_test = np.ones((179, 12), dtype=float)
    new_test_y = read_test[:, 0]
    new_test[:, 1:3] = read_test[:, 0:2]
    new_test[:, 5:9] = read_test[:, 3:7]
    for x in range(0, len(read_test)):
        if read_test[x, 2] == 'male':
            new_test[x, 3] = 1
            new_test[x, 4] = 0
        else:
            new_test[x, 3] = 0
            new_test[x, 4] = 1
        if read_test[x, 7] == 'S':
            new_test[x, 9] = 1
            new_test[x, 10] = 0
            new_test[x, 11] = 0
        elif read_test[x, 7] == 'C':
            new_test[x, 9] = 0
            new_test[x, 10] = 1
            new_test[x, 11] = 0
        else:
            new_test[x, 9] = 0
            new_test[x, 10] = 0
            new_test[x, 11] = 1

    for y in range(1, len(new_train[0])):
        smallest = min(new_train[:, y])
        largest = max(new_train[:, y])
        new_train[:, y] = (new_train[:, y] - smallest) / (largest - smallest)
        smallest_test = min(new_test[:, y])
        largest_test = max(new_test[:, y])
        new_test[:, y] = (new_test[:, y] - smallest_test) / (largest_test - smallest_test)

    learning_rate = 0.01
    w = np.random.normal(0, 0.01, 11)
    max_iters = 1000
    new_train_x = np.zeros((712, 11), dtype=float)
    batch_size = 32
    start = time.time()
    for itr in range(0, max_iters):
        np.random.shuffle(new_train)
        new_train_x[:, 0] = new_train[:, 0]
        new_train_x[:, 1:] = new_train[:, 2:]
        new_train_y = new_train[:, 1]
        sigmoid = 1 / (1 + np.exp((-1) * np.dot(new_train_x[0:batch_size], w.T)))
        w += learning_rate * 1 / batch_size * np.dot(new_train_x[0:batch_size].T,
                                                     (-sigmoid + new_train_y[0:batch_size]))
    end = time.time()
    print("time:", end - start)
    new_test_x = np.zeros((179, 11), dtype=float)
    new_test_x[:, 0] = new_test[:, 0]
    new_test_x[:, 1:] = new_test[:, 2:]
    arr = np.array(1 / (1 + np.exp((-1) * np.dot(new_test_x, w.T))) > 0.5, dtype=int)
    print("Mini batch results")
    print("accuracy: " + str(sum(new_test_y.astype(int) == arr) / 179))
    print("precision: " + str(sum(new_test_y[np.where(arr == 1)].astype(int)) / (sum(arr))))
    true_p = sum(new_test_y[np.where(arr == 1)].astype(int))
    false_p = sum(new_test_y[np.where(arr == 1)].astype(int) == 0)
    true_n = sum(new_test_y[np.where(arr == 0)].astype(int) == 0)
    false_n = (sum(new_test_y.astype(int))) - true_p
    precision_mini = sum(new_test_y[np.where(arr == 1)].astype(int)) / (sum(arr))
    print("recall: " + str(sum(new_test_y[np.where(arr == 1)].astype(int)) / (sum(new_test_y.astype(int)))))
    recall_mini = sum(new_test_y[np.where(arr == 1)].astype(int)) / (sum(new_test_y.astype(int)))
    print("NPV: " + str(sum(new_test_y[np.where(arr == 0)].astype(int) == 0) / (179 - sum(arr))))
    print("FPR: " + str(sum(new_test_y[np.where(arr == 1)].astype(int) == 0) / (179 - sum(new_test_y.astype(int)))))
    print("FDR: " + str(sum(new_test_y[np.where(arr == 1)].astype(int) == 0) / (sum(arr))))
    print("F1: " + str(2 * precision_mini * recall_mini / (precision_mini + recall_mini)))
    print("F2: " + str(5 * precision_mini * recall_mini / (4 * precision_mini + recall_mini)))
    print("TP:", true_p, "FP:", false_p, "FN:", false_n, "TN:", true_n)

    # question_3_2
    print("question 3.2")
    w_full = np.random.normal(0, 0.01, 11)
    print_weight = 0
    np.random.shuffle(new_train)
    start = time.time()
    for itr in range(0, max_iters):
        print_weight += 1
        new_train_x[:, 0] = new_train[:, 0]
        new_train_x[:, 1:] = new_train[:, 2:]
        new_train_y = new_train[:, 1]
        sigmoid = 1 / (1 + np.exp((-1) * np.dot(new_train_x, w_full.T)))
        w_full += learning_rate * 1 / (len(new_train_x)) * np.dot(new_train_x.T, (-sigmoid + new_train_y))
        if print_weight % 100 == 0:
            print("model weights for", print_weight, ":", w_full)
    end = time.time()
    print("time:", end - start)
    new_test_x = np.zeros((179, 11), dtype=float)
    new_test_x[:, 0] = new_test[:, 0]
    new_test_x[:, 1:] = new_test[:, 2:]
    arr = np.array(1 / (1 + np.exp((-1) * np.dot(new_test_x, w_full.T))) > 0.5, dtype=int)
    print("Full batch results")
    print("accuracy: " + str(sum(new_test_y.astype(int) == arr) / 179))
    print("precision: " + str(sum(new_test_y[np.where(arr == 1)].astype(int)) / (sum(arr))))
    precision_full = sum(new_test_y[np.where(arr == 1)].astype(int)) / (sum(arr))
    print("recall: " + str(sum(new_test_y[np.where(arr == 1)].astype(int)) / (sum(new_test_y.astype(int)))))
    recall_full = sum(new_test_y[np.where(arr == 1)].astype(int)) / (sum(new_test_y.astype(int)))
    print("NPV: " + str(sum(new_test_y[np.where(arr == 0)].astype(int) == 0) / (179 - sum(arr))))
    print("FPR: " + str(sum(new_test_y[np.where(arr == 1)].astype(int) == 0) / (179 - sum(new_test_y.astype(int)))))
    print("FDR: " + str(sum(new_test_y[np.where(arr == 1)].astype(int) == 0) / (sum(arr))))
    print("F1: " + str(2 * precision_full * recall_full / (precision_full + recall_full)))
    print("F2: " + str(5 * precision_full * recall_full / (4 * precision_full + recall_full)))
    true_p_full = sum(new_test_y[np.where(arr == 1)].astype(int))
    false_p_full = sum(new_test_y[np.where(arr == 1)].astype(int) == 0)
    true_n_full = sum(new_test_y[np.where(arr == 0)].astype(int) == 0)
    false_n_full = (sum(new_test_y.astype(int))) - true_p
    print("TP:", true_p_full, "FP:", false_p_full, "FN:", false_n_full, "TN:", true_n_full)


# question_4()
def question_4():
    mat = scipy.io.loadmat('q4_dataset.mat')
    temp_feature_set = np.array(mat.get('inception_features'), dtype=float)
    temp_label_set = np.array(mat.get('class_labels'), dtype=float)
    total_feature_set = np.zeros((1250, 2048), dtype=float)
    total_label_set = np.zeros((1250, 1), dtype=float)
    for x in range(0, 5):
        total_feature_set[250 * x:250 * x + 50] = temp_feature_set[50 * x:50 * x + 50]
        total_feature_set[250 * x + 50:250 * x + 100] = temp_feature_set[50 * x + 250:50 * x + 300]
        total_feature_set[250 * x + 100:250 * x + 150] = temp_feature_set[50 * x + 500:50 * x + 550]
        total_feature_set[250 * x + 150:250 * x + 200] = temp_feature_set[50 * x + 750:50 * x + 800]
        total_feature_set[250 * x + 200:250 * x + 250] = temp_feature_set[50 * x + 1000:50 * x + 1050]
        total_label_set[250 * x:250 * x + 50] = temp_label_set[50 * x:50 * x + 50]
        total_label_set[250 * x + 50:250 * x + 100] = temp_label_set[50 * x + 250:50 * x + 300]
        total_label_set[250 * x + 100:250 * x + 150] = temp_label_set[50 * x + 500:50 * x + 550]
        total_label_set[250 * x + 150:250 * x + 200] = temp_label_set[50 * x + 750:50 * x + 800]
        total_label_set[250 * x + 200:250 * x + 250] = temp_label_set[50 * x + 1000:50 * x + 1050]
    for y in range(0, len(total_feature_set[0])):
        smallest = min(total_feature_set[:, y])
        largest = max(total_feature_set[:, y])
        total_feature_set[:, y] = (total_feature_set[:, y] - smallest) / (largest - smallest)
    return total_label_set, total_feature_set


# question_4_1()
def question_4_1():
    start_time = time.time()
    print("question 4.1")
    c_hyper = [10 ** (-6), 10 ** (-4), 10 ** (-2), 1, 10, 10 ** 10]
    select_c = []

    for x in range(0, 5):
        train_set = np.zeros((750, 2048), dtype=float)
        validation_set = np.zeros((250, 2048), dtype=float)
        test_set = np.zeros((250, 2048), dtype=float)
        output_train = np.zeros((750, 1), dtype=float)
        test_set[:, :] = total_feature_set[x * 250:(x + 1) * 250, :]
        validation_set[:, :] = total_feature_set[((x + 1) % 5) * 250:(((x + 1) % 5) + 1) * 250, :]
        output_validation = total_label_set[((x + 1) % 5) * 250:(((x + 1) % 5) + 1) * 250, :]
        if ((x + 1) % 5) > x:
            train_set[0:x * 250, :] = total_feature_set[0:x * 250, :]
            train_set[x * 250:, :] = total_feature_set[(x + 2) * 250:, :]
            output_train[0:x * 250] = total_label_set[0:x * 250, :]
            output_train[x * 250:] = total_label_set[(x + 2) * 250:, :]
        else:
            train_set[:, :] = total_feature_set[250:1000, :]
            output_train[:] = total_label_set[250:1000, :]

        output_train = np.reshape(output_train, (750,))
        output_validation = np.reshape(output_validation, (250,))
        find_max = 0
        find_c = 0
        for c in c_hyper:
            clf = SVC(C=c, kernel='linear')
            clf.fit(train_set, output_train)
            y_test, y_pred = output_validation, clf.predict(validation_set)
            if metrics.precision_score(y_test, y_pred, average='micro') > find_max:
                find_max = metrics.precision_score(y_test, y_pred, average='micro')
                find_c = c
        select_c.append(find_c)

    counter = 0
    for i in select_c:
        curr_frequency = select_c.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            most_freq = i
    results = []
    for x in range(0, 5):
        train_set = np.zeros((1000, 2048), dtype=float)
        test_set = np.zeros((250, 2048), dtype=float)
        output_train = np.zeros((1000, 1), dtype=float)
        test_set[:, :] = total_feature_set[x * 250:(x + 1) * 250, :]
        output_test = total_label_set[x * 250:(x + 1) * 250, :]
        train_set[0:x * 250, :] = total_feature_set[0:x * 250, :]
        train_set[x * 250:, :] = total_feature_set[(x + 1) * 250:, :]
        output_train[0:x * 250] = total_label_set[0:x * 250, :]
        output_train[x * 250:] = total_label_set[(x + 1) * 250:, :]

        output_train = np.reshape(output_train, (1000,))
        clf = SVC(C=most_freq, kernel='linear')
        clf.fit(train_set, output_train)
        y_test, y_pred = output_test, clf.predict(test_set)
        results.append(metrics.precision_score(y_test, y_pred, average='micro'))
        print("Precision:", metrics.precision_score(y_test, y_pred, average='micro'), "C:", most_freq)
    end_time = time.time()
    print("time (in seconds):", end_time - start_time)
    plt.boxplot(results)
    plt.xlabel("linear")
    plt.title("Performance Results")
    plt.show()
    return results


# question_4_2()
def question_4_2():
    start_time = time.time()
    print("question 4.2")
    c_hyper_rbf = [10 ** (-4), 10 ** (-2), 1, 10, 10 ** 10]
    gamma_hyper = [2 ** (-4), 2 ** (-2), 1, 4, 2 ** 10, 'scale']
    select_c = []
    select_kernel = []
    for x in range(0, 5):
        train_set = np.zeros((750, 2048), dtype=float)
        validation_set = np.zeros((250, 2048), dtype=float)
        test_set = np.zeros((250, 2048), dtype=float)
        output_train = np.zeros((750, 1), dtype=float)
        test_set[:, :] = total_feature_set[x * 250:(x + 1) * 250, :]
        validation_set[:, :] = total_feature_set[((x + 1) % 5) * 250:(((x + 1) % 5) + 1) * 250, :]
        output_validation = total_label_set[((x + 1) % 5) * 250:(((x + 1) % 5) + 1) * 250, :]
        if ((x + 1) % 5) > x:
            train_set[0:x * 250, :] = total_feature_set[0:x * 250, :]
            train_set[x * 250:, :] = total_feature_set[(x + 2) * 250:, :]
            output_train[0:x * 250] = total_label_set[0:x * 250, :]
            output_train[x * 250:] = total_label_set[(x + 2) * 250:, :]
        else:
            train_set[:, :] = total_feature_set[250:1000, :]
            output_train[:] = total_label_set[250:1000, :]

        output_train = np.reshape(output_train, (750,))
        output_validation = np.reshape(output_validation, (250,))
        find_max = 0
        find_c = 0
        find_kernel = 0
        for c in c_hyper_rbf:
            for gamma in gamma_hyper:
                clf = SVC(C=c, kernel='rbf', gamma=gamma)
                clf.fit(train_set, output_train)
                y_test, y_pred = output_validation, clf.predict(validation_set)
                if metrics.precision_score(y_test, y_pred, average='micro') > find_max:
                    find_max = metrics.precision_score(y_test, y_pred, average='micro')
                    find_c = c
                    find_kernel = gamma
        select_c.append(find_c)
        select_kernel.append(find_kernel)

    counter = 0
    for i in select_c:
        curr_frequency = select_c.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            most_freq = i
    counter_kernel = 0
    for i in select_kernel:
        curr_frequency_kernel = select_kernel.count(i)
        if curr_frequency_kernel > counter_kernel:
            counter_kernel = curr_frequency_kernel
            most_freq_kernel = i
    results = []
    for x in range(0, 5):
        train_set = np.zeros((1000, 2048), dtype=float)
        test_set = np.zeros((250, 2048), dtype=float)
        output_train = np.zeros((1000, 1), dtype=float)
        test_set[:, :] = total_feature_set[x * 250:(x + 1) * 250, :]
        output_test = total_label_set[x * 250:(x + 1) * 250, :]
        train_set[0:x * 250, :] = total_feature_set[0:x * 250, :]
        train_set[x * 250:, :] = total_feature_set[(x + 1) * 250:, :]
        output_train[0:x * 250] = total_label_set[0:x * 250, :]
        output_train[x * 250:] = total_label_set[(x + 1) * 250:, :]

        output_train = np.reshape(output_train, (1000,))
        clf = SVC(C=most_freq, kernel='rbf', gamma=most_freq_kernel)
        clf.fit(train_set, output_train)
        y_test, y_pred = output_test, clf.predict(test_set)
        results.append(metrics.precision_score(y_test, y_pred, average='micro'))
        print("Precision:", metrics.precision_score(y_test, y_pred, average='micro'), "C:", most_freq, "gamma:",
              most_freq_kernel)
    end_time = time.time()
    print("total time (in seconds):", end_time - start_time)
    plt.boxplot(results)
    plt.xlabel("rbf")
    plt.title("Performance Results")
    plt.show()
    return results


# running part
# question 1
inp = preprocess()
question_1_1(inp)

pimp = preprocess()
question_1_2(pimp)

# question 2
mse_linear, mae_linear, r_square_linear, mape_linear = question_2_2()
mse_lasso, mae_lasso, r_square_lasso, mape_lasso = question_2_3()
results = {'linear': r_square_linear, 'lasso': r_square_lasso}
fig, ax = plt.subplots()
ax.boxplot(results.values())
ax.set_xticklabels(results.keys())
plt.title("Performance Results for R^2")
plt.show()

results = {'linear': mse_linear, 'lasso': mse_lasso}
fig, ax = plt.subplots()
ax.boxplot(results.values())
ax.set_xticklabels(results.keys())
plt.title("Performance Results for MSE")
plt.show()

results = {'linear': mae_linear, 'lasso': mae_lasso}
fig, ax = plt.subplots()
ax.boxplot(results.values())
ax.set_xticklabels(results.keys())
plt.title("Performance Results for MAE")
plt.show()

results = {'linear': mape_linear, 'lasso': mape_lasso}
fig, ax = plt.subplots()
ax.boxplot(results.values())
ax.set_xticklabels(results.keys())
plt.title("Performance Results for MAPE")
plt.show()

# question 3
question_3_1_and_3_2()

# question 4
total_label_set, total_feature_set = question_4()
result_linear = question_4_1()
result_rbf = question_4_2()
results = {'linear': result_linear, 'rbf': result_rbf}
fig, ax = plt.subplots()
ax.boxplot(results.values())
ax.set_xticklabels(results.keys())
plt.title("Performance Results")
plt.show()
