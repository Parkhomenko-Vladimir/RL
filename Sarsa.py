import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, LSTM,SimpleRNNCell, Reshape,TimeDistributed
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import random
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow import keras

class SARSA:

    def __init__(self, input_dimention, num_actions, gamma, max_experiences, min_experiences, batch_size, lr):
        """
        :type self.net = нейронка
        :type self.experience = буффер
        :type self.min_experiences: Минимальное значение буфера
        :type self.max_experiences: Максимальное значение буфера(при заполнении очищается полностью)
        :type self.gamma: Коэфициент дисконтирования
        :type lr: Скорость обучения
        :type self.batch_size: Размер минивыборки для обучения
        :type self.num_actions: количество выходов НС
        """
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.SGD(lr)
        self.gamma = gamma

        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'a2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences  

        self.net = MyModel(input_dimention, [200, 200], self.num_actions)

        # inp_shape = Input( shape=input_dimention )
        # hd = Dense(input_dimention, activation='sigmoid')(inp_shape)
        # hd1 = Dense(15, activation='sigmoid')(hd)
        # hd2 = Dense(16, activation='sigmoid')(hd1)
        # drop = Dropout(0.3)(hd2)
        # hd3 = Dense(64, activation='sigmoid')(drop)
        # hd4 = Dense(32, activation='sigmoid')(hd3)
        #
        # Out = Dense(self.num_actions, activation='sigmoid')(hd4)
        #
        # self.net = Model(inputs=inp_shape, outputs=Out)

        # self.net.compile(optimizer='sgd',
        #               loss='sparse_categorical_crossentropy',
        #               metrics=['accuracy'])

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
        action_next = np.asarray([self.experience['a2'].pop(0) for i in range(self.batch_size)])
        dones =       np.asarray([self.experience['done'].pop(0) for i in range(self.batch_size)])

        value_next = self.predict(states_next)

        q_actions =  np.asarray([value_next[i][action_next[i]] for i in range(self.batch_size)])

        actual_values = np.where(dones, rewards, rewards + self.gamma * q_actions)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def train(self):
        '''
        Метод обучения нейронки
        :return: loss (значение потерь)
        '''
        if len(self.experience['s']) < self.min_experiences:
            return 0

        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)

        states =      np.asarray([self.experience['s'][i] for i in ids])
        actions =     np.asarray([self.experience['a'][i] for i in ids])
        rewards =     np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        action_next = np.asarray([self.experience['a2'][i] for i in ids])
        dones =       np.asarray([self.experience['done'][i] for i in ids])

        value_next = self.predict(states_next)

        q_actions =  np.asarray([value_next[i][action_next[i]] for i in range(self.batch_size)])

        actual_values = np.where(dones, rewards, rewards + self.gamma * q_actions)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def predict(self, inputs):
        '''
        Метод распознавания картинки сетью
        :param inputs: Состояние (или состояния) среды, подаваемое на картинку
        :return: возвращает массив вероятностей для каждого действия
        '''
        return self.net(np.atleast_2d(inputs.astype('float32')))

    def get_action(self, state, epsilon):
        '''
        Метод e-greedy принятия решений
        :param epsilon: Вероятность принятия случайного решения
        :param state: Состояние среды, подаваемое на картинку
        :return: возвращает номер выбранного действия
        '''
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.net( np.atleast_2d(state.astype('float32')) )[0])

    def add_experience(self, exp):
        '''
        Метод добавления сыгранного шага в буфер. Если буффер переполнен, то производится очистка
        :param батч из S A R S A Done
        :return: возвращает обновленное состояние буфера
        '''
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output