import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
import random
from tensorflow.keras import layers

class DDQN:

    def __init__(self, input_dimention, num_actions, gamma, max_experiences, min_experiences, batch_size, lr, load_net_path=None):
        """
        :type self.net = нейронка
        :type self.experience = буффер
        :type self.min_experiences: Минимальное значение буфера
        :type self.max_experiences: Максимальное значение буфера(при заполнении очищается полностью)
        :type self.gamma: Коэфициент дисконтирования
        :type self.state_size: Размерность состояния среды
        :type lr: Скорость обучения
        :type self.batch_size: Размер минивыборки для обучения
        :type self.num_actions: количество выходов НС
        """
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.state_size = input_dimention

        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

        self.Target_net = Resnet(shape=input_dimention,
                                 num_actions=self.num_actions,
                                 num_bloks=9)
        self.Train_net = Resnet(shape=input_dimention,
                                num_actions=self.num_actions,
                                num_bloks=9)


    def train(self):
        '''
        Метод обучения нейронки вторая версия(от первого отличается синтаксисом)
        :return: tensorflow.python.keras.callbacks.History object
        '''
        if len(self.experience['s']) < self.min_experiences:
            return 0

        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        
        states =      np.asarray([self.experience['s'][i] for i in ids])
        actions =     np.asarray([self.experience['a'][i] for i in ids])
        rewards =     np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones =       np.asarray([self.experience['done'][i] for i in ids])

        if np.random.random() < 0.5:

            value_A = np.argmax(self.Target_net_predict(states_next), axis=1)
            value_b = self.Train_net_predict(states_next)

            # actual_values = [rewards[i] + self.gamma * value_b[i][value_A[i]] for i in range(self.batch_size)]

            q_next = np.asarray([value_b[i][value_A[i]] for i in range(self.batch_size)])
            actual_values = np.where(dones, rewards, rewards + self.gamma * q_next)

            with tf.GradientTape() as tape:
                selected_action_values = tf.math.reduce_sum(
                    self.Target_net_predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
                loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
            variables = self.Target_net.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            return loss
        else:
            value_A = np.argmax(self.Train_net_predict(states_next), axis=1)
            value_b = self.Target_net_predict(states_next)

            # actual_values = [rewards[i] + self.gamma * value_b[i][value_A[i]] for i in range(self.batch_size)]

            q_next = np.asarray([value_b[i][value_A[i]] for i in range(self.batch_size)])
            actual_values = np.where(dones, rewards, rewards + self.gamma * q_next)

            with tf.GradientTape() as tape:
                selected_action_values = tf.math.reduce_sum(
                    self.Train_net_predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
                loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
            variables = self.Train_net.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            return loss

    def train_pop(self):
        '''
        Метод обучения нейронки
        :return: loss (значение потерь)
        '''
        if len(self.experience['s']) < self.min_experiences:
            return 0

        states =      np.asarray([self.experience['s'].pop(0) for i in range(self.batch_size)])
        actions =     np.asarray([self.experience['a'].pop(0) for i in range(self.batch_size)])
        rewards =     np.asarray([self.experience['r'].pop(0) for i in range(self.batch_size)])
        states_next = np.asarray([self.experience['s2'].pop(0) for i in range(self.batch_size)])
        dones =       np.asarray([self.experience['done'].pop(0) for i in range(self.batch_size)])

        if np.random.random() < 0.5:

            value_A = np.argmax(self.Target_net_predict(states_next), axis=1)
            value_b = self.Train_net_predict(states_next)

            # actual_values = [rewards[i] + self.gamma * value_b[i][value_A[i]] for i in range(self.batch_size)]

            q_next = np.asarray([value_b[i][value_A[i]] for i in range(self.batch_size)])
            actual_values = np.where(dones, rewards, rewards + self.gamma * q_next)

            with tf.GradientTape() as tape:
                selected_action_values = tf.math.reduce_sum(
                    self.Target_net_predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
                loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
            variables = self.Target_net.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            return loss
        else:
            value_A = np.argmax(self.Train_net_predict(states_next), axis=1)
            value_b = self.Target_net_predict(states_next)

            # actual_values = [rewards[i] + self.gamma * value_b[i][value_A[i]] for i in range(self.batch_size)]

            q_next = np.asarray([value_b[i][value_A[i]] for i in range(self.batch_size)])
            actual_values = np.where(dones, rewards, rewards + self.gamma * q_next)

            with tf.GradientTape() as tape:
                selected_action_values = tf.math.reduce_sum(
                    self.Train_net_predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
                loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
            variables = self.Train_net.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            return loss


    def add_experience(self, exp):
        '''
        Метод добавления сыгранного шага в буфер. Если буффер переполнен, то производится очистка
        :param батч из S A R S
        :return: возвращает обновленное состояние буфера
        '''
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
    
        for key, value in exp.items():
            self.experience[key].append(value)

    def Target_net_predict(self, inputs):
        '''
        Метод распознавания картинки Target - сетью
        :param inputs: Состояние (или состояния) среды, подаваемое на картинку
        :return: возвращает массив вероятностей для каждого действия
        '''
        return self.Target_net(np.atleast_2d(inputs.astype('float32')))

    def Train_net_predict(self, inputs):
        '''
        Метод распознавания картинки Train - сетью
        :param inputs: Состояние (или состояния) среды, подаваемое на картинку
        :return: возвращает массив вероятностей для каждого действия
        '''
        return self.Train_net(np.atleast_2d(inputs.astype('float32')))

    def get_action(self, states, epsilon):
        '''
        Метод e-greedy принятия решений
        :param epsilon: Вероятность принятия случайного решения
        :param states: Состояние среды, подаваемое на картинку
        :return: возвращает номер выбранного действия
        '''
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.Target_net(np.atleast_2d(states.astype('float32')))[0])

    def save_train_net(self, name):
        '''
        Сохранение train нейронки
        name - string
        '''
        self.Train_net.save_weights( name + '.h5' )

    def save_target_net(self, name):
        '''
        Сохранение target нейронки
        name - string
        '''
        self.Target_net.save_weights(name + '.h5')

    def load_train_net(self,path):
        '''
        Загрузка весов train нейронки
        path - полный путь
        '''
        self.state_size[0]
        self.Train_net.build((1, self.state_size[0],
                                 self.state_size[1],
                                 self.state_size[2]))
        self.Train_net.load_weights(path)

    def load_target_net(self,path):
        '''
        Загрузка весов target нейронки
        path - полный путь
        '''
        self.Target_net.build((1, self.state_size[0],
                                  self.state_size[1],
                                  self.state_size[2]))
        self.Target_net.load_weights(path)


class ResBlok(tf.keras.Model):
    def __init__(self, n_filters, kernel_size):
        super(ResBlok, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.b1 = layers.Conv2D(filters=self.n_filters, kernel_size=self.kernel_size,
                                activation='relu', padding='same')
        self.b2 = layers.BatchNormalization()
        self.b3 = layers.Conv2D(filters=self.n_filters, kernel_size=self.kernel_size,
                                activation=None, padding='same')
        self.b4 = layers.BatchNormalization()
        self.b5 = layers.Add()
        self.b6 = layers.Activation('relu')

    def call(self, inp):
        x = self.b1(inp)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5([inp, x])
        y = self.b6(x)

        return y

class Resnet(tf.keras.Model):
    def __init__(self, shape, num_actions, num_bloks):
        super(Resnet, self).__init__()

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(shape))

        self.s1 = layers.Conv2D(32, 2, activation='relu')
        self.s2 = layers.Conv2D(64, 2, activation='relu')
        self.s3 = layers.MaxPooling2D(2)

        self.list_res_bl = tf.keras.Sequential()
        for i in range(num_bloks):
            self.list_res_bl.add(ResBlok(n_filters=64, kernel_size=2))

        self.end1 = layers.Conv2D(64, 2, activation='relu')
        self.end2 = layers.GlobalAveragePooling2D()
        self.end3 = layers.Dense(256, activation='relu')
        self.end4 = layers.Dropout(0.5)

        self.out = layers.Dense(num_actions, activation='softmax')

    def call(self, inp):
        x = self.s1(inp)
        x = self.s2(x)
        x = self.s3(x)

        x = self.list_res_bl(x)

        x = self.end1(x)
        x = self.end2(x)
        x = self.end3(x)
        x = self.end4(x)
        outp = self.out(x)

        return outp
