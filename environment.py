import numpy as np

import matplotlib.pyplot as plt
from random import randint
from math import floor
import random

class Env():
    '''
    Класс среды для Rl агента. Генерирует карту с препятствиями, начальные и целевые координаты, позволяетт совершать агенту действия в среде.
    '''

    def __init__(self, dim, countOBS, sizeOBS, str_x_y=None, trg_x_y=None, seed=None,
                 gamma=0.95,learning=False):
        '''
        :type seed: зерно для генерации
        :param dim: размерность карты
        :param countOBS: максимальное количество препятствий + 1
        :param sizeOBS: размер препятствий (0-100)
        :param st_x: начальное продольное положение
        :param st_y: начальное поперечное положение
        :param str_x_y: изначально заданное начальное положение
        :param trg_x_y: изначально заданное целевое положение
        :param self.x_dim - размерность карты (х - ось)
        :param self.y_dim - размерность карты (у - ось)
        :param self.done = False - игра не окончена, True - игра окончена

        :param self.step_r - реворд за шаг
        :param self.obs_r - реворд за препятствие
        :param self.board_r - реворд за границы
        :param self.max_r - реворд за цель
        :param self.sum_r - сумма ревордов за сессию
        :param self.sum_limit - значение суммы ревордов, при достижении которой останавливаем сессию

        :param self.delta - коэфицент, на который уножается разность потенциалов
        :param self.pnt_dict - словарь для преобразования выбранного направления агентом в координаты на матрице
        :param learning - если True - закрываем график после инициализации среды,
                               False - не закрываем
        '''

        self.seed = seed
        self.x_dim = dim[0]  # размерность карты х - ось
        self.y_dim = dim[1]   # размерность карты у - ось
        self.done = False  # False - игра не окончена, True - игра окончена
        self.gamma = gamma #Коэфицент дисконтирования

        # self.step_r = 0  # реворд за шаг
        self.obs_r = -10  # реворд за препятствие
        self.board_r = -10  # реворд за границы
        self.max_r = 10  # реворд за цель
        self.sum_r = 0
        self.sum_limit = -10 #  !!!Должно быть отрицательным числом!!!

        self.delta = 0.1

        self.countOBS = countOBS  # countOBS - максимальное количество препятствий + 1
        self.sizeOBS = sizeOBS  # sizeOBS - размер препятствий (0-100)

        self.pnt_dict = {
            str(0): [1, 0],
            str(1): [1, -1],
            str(2): [0, -1],
            str(3): [-1, -1],
            str(4): [-1, 0],
            str(5): [-1, 1],
            str(6): [0, 1],
            str(7): [1, 1]
        }

#         self.pnt_dict = {
#             str(0): [1, 0],
#             str(1): [0, -1],
#             str(2): [-1, 0],
#             str(3): [0, 1]
#         }

        random.seed(self.seed) # зерно для генерации

        self.create_random_obstacles(countOBS=self.countOBS,
                                     sizeOBS=self.sizeOBS,
                                     mymap=np.zeros((self.y_dim, self.x_dim)))  # Готовая карта с препятствиями

        if str_x_y == None or trg_x_y == None:
            self.gen_points()
        else:
            self.st_x = str_x_y[0]  # Начальные координаты
            self.st_y = str_x_y[1] + 1# Начальные координаты
            self.cur_x, self.cur_y = self.st_x, self.st_y

            self.trg_x = trg_x_y[0]  # Целевые координаты
            self.trg_y = trg_x_y[1] + 1# Целевые координаты

        assert (not self.map_matrix[self.y_dim - self.trg_y , self.trg_x]), 'Env():__init__:trg_x_y: point on obstacle'
        assert (not self.map_matrix[self.y_dim - self.st_y , self.st_x]), 'Env():__init__:st_x_y: point on obstacle'

        self.update_map()

        if learning == True:
            plt.close()
            plt.pause(1.5)

    def update_scene(self,d=None):
        '''
        Полное обновление карты с препятствиями и точками
        d  = дистанция между точками
        '''
        self.create_random_obstacles(countOBS=self.countOBS,
                                     sizeOBS=self.sizeOBS,
                                     mymap=np.zeros((self.y_dim, self.x_dim)))

        self.gen_points(d)
        # self.learning_restart_this_game()

    def special_target(self,r):
        '''
        Метод генерации целевой точки в центральной области
        :return: возвращает обновленные значения self.trg_x, self.trg_y, self.st_x, self.st_y
        '''
        if r <= self.x_dim / 2:
            self.trg_x, self.trg_y = random.randint(self.x_dim//2 - r - 1, self.x_dim//2 + r - 1),\
                                     random.randint(self.x_dim//2 - r,     self.x_dim//2 + r)
            while (self.map_matrix[self.y_dim - self.trg_y, self.trg_x]):
                self.trg_x, self.trg_y = random.randint(self.x_dim // 2 - r - 1, self.x_dim // 2 + r - 1),\
                                         random.randint(self.x_dim // 2 - r,     self.x_dim // 2 + r)
        else:
            self.trg_x, self.trg_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim)



        self.st_x, self.st_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim)
        while (self.map_matrix[self.y_dim - self.st_y, self.st_x] or \
               (self.trg_y == self.st_y and self.trg_x == self.st_x)):
            self.st_x, self.st_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim)

        self.cur_x, self.cur_y = self.st_x, self.st_y

    def gen_points(self, t_dist = None):
        '''
        Метод генерации точек старта и цели на заданном расстоянии
        :param t_dist: максимальное расстояние между стартом и целью. Если t_dist != None - растояние будет любым
        :return: возвращает обновленные значения self.st_x, self.st_y
        '''

        if t_dist == None:

            self.st_x, self.st_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim )
            while (self.map_matrix[self.y_dim - self.st_y, self.st_x]):

                self.st_x, self.st_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim )

            self.trg_x, self.trg_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim )
            while (self.map_matrix[self.y_dim - self.trg_y , self.trg_x] or
                   (self.trg_y == self.st_y and self.trg_x == self.st_x)):

                self.st_x, self.st_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim )

        else:

            self.st_x, self.st_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim)

            while (self.map_matrix[self.y_dim - self.st_y, self.st_x]):
                self.st_x, self.st_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim)

            self.trg_x, self.trg_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim)
            space = np.sqrt((self.trg_x - self.st_x) ** 2 + (self.trg_y - self.st_y) ** 2)

            while (self.map_matrix[self.y_dim - self.trg_y , self.trg_x] or  \
                   (self.trg_y == self.st_y and self.trg_x == self.st_x)) or \
                   t_dist < space:

                self.trg_x, self.trg_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim)
                space = np.sqrt((self.trg_x - self.st_x) ** 2 + (self.trg_y - self.st_y) ** 2)

        self.cur_x, self.cur_y = self.st_x, self.st_y

    def gen_start_points(self, t_dist = None):
        '''
        Метод генерации ТОЛЬКО начальных положений
        :param t_dist: максимальное расстояние между стартом и целью. Если t_dist != None - растояние будет любым
        :return: возвращает обновленные значения self.st_x, self.st_y
        '''

        if t_dist == None:

            self.st_x, self.st_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim )
            while (self.map_matrix[self.y_dim - self.trg_y , self.trg_x] or
                   self.map_matrix[self.y_dim - self.st_y, self.st_x] or
                   (self.trg_y == self.st_y and self.trg_x == self.st_x)):

                self.st_x, self.st_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim )

        else:

            self.st_x, self.st_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim )
            space = np.sqrt((self.trg_x - self.st_x) ** 2 + (self.trg_y - self.st_y) ** 2)

            while (self.map_matrix[self.y_dim - self.trg_y , self.trg_x] or
                   self.map_matrix[self.y_dim - self.st_y, self.st_x] or
                   (self.trg_y == self.st_y and self.trg_x == self.st_x)) or \
                   t_dist < space:

                self.st_x, self.st_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim )
                space = np.sqrt((self.trg_x - self.st_x) ** 2 + (self.trg_y - self.st_y) ** 2)

        self.cur_x, self.cur_y = self.st_x, self.st_y

    def gen_target_points(self, t_dist = None):
        '''
        Метод генерации ТОЛЬКО целевых положений
        :param t_dist: максимальное расстояние между стартом и целью. Если t_dist != None - растояние будет любым
        :return: возвращает обновленные значения self.st_x, self.st_y
        '''

        if t_dist == None:

            self.trg_x, self.trg_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim )
            while (self.map_matrix[self.y_dim - self.trg_y , self.trg_x] or
                   self.map_matrix[self.y_dim - self.st_y, self.st_x] or
                   (self.trg_y == self.st_y and self.trg_x == self.st_x)):

                self.trg_x, self.trg_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim )

        else:

            self.trg_x, self.trg_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim )
            space = np.sqrt((self.trg_x - self.st_x) ** 2 + (self.trg_y - self.st_y) ** 2)

            while (self.map_matrix[self.y_dim - self.trg_y , self.trg_x] or
                   self.map_matrix[self.y_dim - self.st_y, self.st_x] or
                   (self.trg_y == self.st_y and self.trg_x == self.st_x)) or \
                   t_dist < space:

                self.trg_x, self.trg_y = random.randint(0, self.x_dim - 1), random.randint(1, self.y_dim )
                space = np.sqrt((self.trg_x - self.st_x) ** 2 + (self.trg_y - self.st_y) ** 2)

        self.cur_x, self.cur_y = self.st_x, self.st_y

    def create_random_obstacles(self, countOBS, sizeOBS, mymap, num_to_fill=1):
        '''
        Создает карту препятствий, пердставленную матрицей.

        :param countOBS: максимальное количество генерируемых препятствий - 1
        :param sizeOBS: доля заполнения карты препятствиями (1 - 100)
        :param mymap: матрица препятствий (0 - свободная клетка, 1 - клетка с препятствием)
        :param num_to_fill: - по умолчанию 1
        :return: матрица препятствий (0 - свободная клетка, 1 - клетка с препятствием)
        '''

        size_map_0, size_map_1 = mymap.shape
        OBS_max_size = floor((size_map_0 + size_map_1 - 2) * (sizeOBS / 100.))
        OBS_min_size = floor((OBS_max_size / 3.))

        obstacle_params = []
        for i in range(countOBS):
            count_joint = randint(1, 4)
            coords = [randint(0, size_map_0 - 1), randint(0, size_map_1 - 1)]

            joints_params = []
            for j in range(count_joint):
                length_cell = randint(OBS_min_size, OBS_max_size)
                width_cell = randint(OBS_min_size, OBS_max_size)
                orientation = randint(1, 4)

                new_x = coords[0]
                new_y = coords[1]

                dict_orientation = {'1': [new_x + width_cell, new_y + length_cell],
                                    '2': [new_x + width_cell, new_y - length_cell],
                                    '3': [new_x - length_cell, new_y + width_cell],
                                    '4': [new_x + length_cell, new_y + width_cell]}
                new_x, new_y = dict_orientation[str(orientation)]

                if new_x > (size_map_0 - 1):
                    new_x = size_map_0 - 1
                if new_x < 0:
                    new_x = 0
                if new_y > (size_map_1 - 1):
                    new_y = size_map_1 - 1
                if new_y < 0:
                    new_y = 0

                if coords[0] > new_x:
                    coords[0], new_x = new_x, coords[0]
                if coords[1] > new_y:
                    coords[1], new_y = new_y, coords[1]

                joints_params.append([coords[0], coords[1], new_x, new_y, orientation])
                coords = [new_x, new_y]

            obstacle_params.append(joints_params)

        for i in range(len(obstacle_params)):
            for j in range(len(obstacle_params[i])):
                joints_params = obstacle_params[i][j]

                mymap[joints_params[0]:joints_params[2], joints_params[1]:joints_params[3]] = num_to_fill

        self.map_matrix = mymap

    def step(self, direction):
        '''
        Позволяет осеществлять действие в среде.
        :param direction: направление движения, относительно настоящего положения (0-7)
        :return: возвращает спиок [self.img, self.board_r, self.sum_r, self.done] - [трехслойная матрица, реворд, суммарный реворд, закончена игра или нет]
        '''
        img, r, sum_r, done = self.learning_step(direction)
        self.update_map()

        return (img, r, sum_r, done)

    def update_map(self):
        '''
        Позволяет выводить текущее состояние среды в окне. Обновляет текущее состояние среды.
        :return: Обновляет текущее состояние среды на графике
        '''
        img = np.zeros((self.y_dim, self.x_dim, 3), dtype=np.int32)

        img[self.y_dim - self.trg_y , self.trg_x, 0] = 1.  # Целевое положение
        img[:, :, 1] = self.map_matrix  # Карта
        img[self.y_dim - self.cur_y , self.cur_x, 2] = 1.  # Начальное положение
        self.img = img

        plt.imshow((self.img * 255.).astype(np.uint8))
        plt.pause(0.5)

    def restart_this_game(self):
        """
        Начинает игры заново, с теми же начальными условиями
        :return: Обновляет параметры: done, текущее положение, 3-мерное изображение, сумма ревордов за сессию
        """
        self.done = False
        self.cur_x, self.cur_y = self.st_x, self.st_y
        self.sum_r = 0
        self.update_map()

    def learning_step(self, direction):
        '''
        Позволяет осеществлять действие в среде. Полностю тоже самое, что и self.step, только не выводит графическую часть.
        :param direction: направление движения, относительно настоящего положения (0-7)
        :param r: - значение разницы потенциалов между стартом и целью
        :return: возвращает спиок [self.img, self.board_r, self.sum_r, self.done] - [трехслойная матрица, реворд, суммарный реворд, закончена игра или нет]
        '''
        # print(str(direction))
        pnt = self.pnt_dict[str(direction)]

        # Условие выхода за границу карты
        if not ( 0 <= self.cur_x + pnt[0] <= self.x_dim-1 ) or not ( 1 <= self.cur_y + pnt[1] <= self.y_dim ):
            # print("Not avaliable to aboard")

            self.sum_r = self.sum_r + self.board_r
            # условие на сумму реворда
            if self.sum_r <= self.sum_limit:
                self.done = True

            return ([self.img, self.board_r, self.sum_r, self.done])

        # Условие колиизии
        if self.map_matrix[self.y_dim  - self.cur_y - pnt[1], self.cur_x + pnt[0]]:

            self.sum_r = self.sum_r + self.obs_r
            # условие на сумму реворда
            if self.sum_r <= self.sum_limit:
                self.done = True

            return ([self.img, self.obs_r, self.sum_r, self.done])

        # Условие достижения цели (завершение игры)
        if self.cur_x + pnt[0] == self.trg_x and self.cur_y + pnt[1] == self.trg_y:
            self.cur_x = self.cur_x + pnt[0]
            self.cur_y = self.cur_y + pnt[1]
            self.sum_r = self.sum_r + self.max_r
            self.done = True

            # Обновление матрицы
            img = np.zeros((self.y_dim, self.x_dim, 3), dtype=np.int32)
            img[self.y_dim - self.trg_y , self.trg_x, 0] = 10.  # Целевое положение
            img[:, :, 1] = self.map_matrix  # Карта
            img[self.y_dim - self.cur_y , self.cur_x, 2] = 10.  # Начальное положение

            # print('end of game')
            return (self.img, self.max_r, self.sum_r, self.done)

        # При невыполнении вышеописанных условий, осушествляется шаг в среде

        prev_dist = np.sqrt( (self.trg_x - self.cur_x)**2 + (self.trg_y - self.cur_y)**2 )
        new_dist =  np.sqrt( (self.trg_x - (self.cur_x + pnt[0])) ** 2 +
                             (self.trg_y - (self.cur_y + pnt[1])) ** 2) + 0.00001
        if new_dist<prev_dist:
            r = self.delta * np.abs((self.gamma * new_dist) - prev_dist)
        else:
            r = - self.delta * np.abs((self.gamma * new_dist) - prev_dist)

        self.cur_x = self.cur_x + pnt[0]
        self.cur_y = self.cur_y + pnt[1]

        self.sum_r = self.sum_r  + r

        # Обновление матрицы
        img = np.zeros((self.y_dim, self.x_dim, 3), dtype=np.int32)
        img[self.y_dim - self.trg_y , self.trg_x, 0] = 10.  # Целевое положение
        img[:, :, 1] = self.map_matrix  # Карта
        img[self.y_dim - self.cur_y , self.cur_x, 2] = 10.  # Начальное положение

        self.img = img

        # условие на сумму реворда
        if self.sum_r <= self.sum_limit:
            self.done = True

        return (self.img, r , self.sum_r, self.done)

    def learning_restart_this_game(self):
        """
        Начинает игру заново, с теми же начальными условиями. Тоже самое, что и learning_step, только и без графической части
        :return: Обновляет параметры: done, текущее положение, 3-мерное изображение, сумма ревордов за сессию
        """
        self.done = False
        self.cur_x, self.cur_y = self.st_x, self.st_y
        self.sum_r = 0

        # Обновление матрицы
        img = np.zeros((self.y_dim, self.x_dim, 3), dtype=np.int32)
        img[self.y_dim - self.trg_y , self.trg_x, 0] = 10.  # Целевое положение
        img[:, :, 1] = self.map_matrix  # Карта
        img[self.y_dim - self.cur_y , self.cur_x, 2] = 10.  # Начальное положение

        self.img = img