import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import random
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow import keras

class EVSARSA:

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
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma

        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

        self.net = MyModel(input_dimention, [200, 200], self.num_actions)

    def get_target_value(self,Qs1,epsilon):

        # Q_max = np.max(Qs1)
        # numb_Q_max = np.argmax(Qs1)

        trans_prob = 0
        state_value =0

        for act in Qs1:

            if act == np.max(Qs1):
                trans_prob = 1.0 - epsilon + epsilon / len(Qs1)
                # trans_prob = 1.0 - epsilon

            else:
                trans_prob = epsilon / len(Qs1)
                # trans_prob = epsilon / (self.num_actions - 1 )


            state_value = state_value + trans_prob * act

        return state_value


    def train(self,eps):
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
        dones =       np.asarray([self.experience['done'][i] for i in ids])

        Q_t =           np.asarray(self.predict(states_next))
        value_next =    np.asarray([self.get_target_value(i, eps) for i in Q_t])
        actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def train_pop(self,eps):
        '''
        Метод обучения нейронки вторая версия(от первого отличается синтаксисом)
        :return: tensorflow.python.keras.callbacks.History object
        '''
        if len(self.experience['s']) < self.min_experiences:
            return 0

        states =      np.asarray([self.experience['s'].pop(0) for i in range(self.batch_size)])
        actions =     np.asarray([self.experience['a'].pop(0) for i in range(self.batch_size)])
        rewards =     np.asarray([self.experience['r'].pop(0) for i in range(self.batch_size)])
        states_next = np.asarray([self.experience['s2'].pop(0) for i in range(self.batch_size)])
        dones =       np.asarray([self.experience['done'].pop(0) for i in range(self.batch_size)])

        Q_t = np.asarray(self.predict(states_next))

        value_next =  np.asarray([self.get_target_value(i, eps) for i in Q_t])

        actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss


    def add_experience(self, exp):
        '''
        Метод добавления сыгранного шага в буфер. Если буффер переполнен, то производится очистка
        :param батч из S A R S  Done
        :return: возвращает обновленное состояние буфера
        '''
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

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