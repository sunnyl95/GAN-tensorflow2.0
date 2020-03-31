import math
import numpy as np
import tensorflow as tf
import datetime
from PIL import Image
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Dense, Activation, BatchNormalization, Reshape, Conv2D, UpSampling2D, \
    MaxPooling2D, Flatten


def generator_model():
    model = Sequential()

    model.add(Dense(1024, input_dim=100))
    model.add(Activation('tanh'))

    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())  # 批量归一化: 将前一层的激活值重新规范化，使得其输出数据的均值接近0，其标准差接近1
    model.add(Activation('tanh'))

    model.add(Reshape((7, 7, 128)))

    # 2维上采样层，即将数据的行和列分别重复2次
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))

    model.add(UpSampling2D(size=(2, 2)))

    # 卷积核设为1即输出图像的维度
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('tanh'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    # 先添加生成器架构，再令d不可训练，即固定d
    # 因此在给定d的情况下训练生成器，即通过将生成的结果投入到判别器进行辨别而优化生成器
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    # 生成图片拼接
    num = generated_images.shape[0]
    width = int(math.ceil(math.sqrt(num)))
    height = int(math.ceil(float(num) / width))  # math.ceil 取天棚
    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img[:, :, 0]
    return image


def write_log(writer, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.compat.v1.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        writer.add_summary(summary, batch_no)
        writer.flush()


def train(batch_size):
    (x_train, y_train), (_, _) = mnist.load_data()
    # x_train = x_train[y_train == 0]
    print(x_train.shape)

    # 转换字段类型，并将数据导入变量中
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = x_train[:, :, :, None]  # None将3维的X_train扩展为4维

    # 将定义好的模型架构赋值给特定的变量
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)

    # 定义生成器模型判别器模型更新所使用的优化算法及超参数
    d_optim = SGD(lr=0.001, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.001, momentum=0.9, nesterov=True)

    # 编译三个神经网络并设置损失函数和优化算法，其中损失函数都是用的是二元分类交叉熵函数。编译是用来配置模型学习过程的
    g.compile(loss='binary_crossentropy', optimizer='SGD')
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)

    # 前一个架构在固定判别器的情况下训练了生成器，所以在训练判别器之前先要设定其为可训练。
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    # 使用tensorboard跟踪数据
    current_time = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    log_dir = 'logs/' + current_time
    writer = tf.compat.v1.summary.FileWriter(log_dir)

    for epoch in range(30):
        batches_num = int(x_train.shape[0] / batch_size)  # 取地板
        for index in range(int(x_train.shape[0] / batch_size)):
            if index % 10 == 0:
                print('Epoch: %d %d/%d' % (epoch, index, batches_num))
            noise = np.random.uniform(-1, 1, size=(batch_size, 100))  # -1~1 的均匀分布
            image_batch = x_train[index * batch_size:(index + 1) * batch_size]  # 抽取一个批量的真实图片
            generated_images = g.predict(noise, verbose=0)  # 根据随机噪声生成图片 shape: (batch_size,28,28,1)

            if index == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save('./GAN/' + str(epoch) + '.png')
                print('Combine image created..')

            x = np.concatenate((image_batch, generated_images))  # 将真实图片和生成图片放到一起 shape: (2*batch_size,28,28,1)
            y = [1] * batch_size + [0] * batch_size  # 真实图片标签为1， 生成图片标签为0

            d_loss = d.train_on_batch(x, y)  # 训练 d
            # print('batch %d d_loss : %f' % (index, d_loss))

            d.trainable = False
            noise = np.random.uniform(-1, 1, (batch_size, 100))
            y = [1] * batch_size
            g_loss = d_on_g.train_on_batch(noise, y)  # g的目标是愚弄辨别器蒙混过关，使对于生成的图片输出为1
            d.trainable = True
            # print('batch %d g_loss : %f' % (index, g_loss))

            write_log(writer, ['d_loss', 'g_loss'], [d_loss, g_loss], epoch * batches_num + index)


train(100)
