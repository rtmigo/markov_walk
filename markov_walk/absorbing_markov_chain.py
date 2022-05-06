# SPDX-FileCopyrightText: (c) 2018 Art Galkin <ortemeo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

# If you've ever wondered what comments look like in Russian, scroll down.

import unittest
from functools import cached_property  # needs python 3.8

import numpy as np
from numpy import ndarray


def ensure_stochastic_matrix(P: ndarray) -> ndarray:
    # убеждаемся, что матрица квадратная
    if P.shape[0] != P.shape[1]:
        raise ValueError("The matrix shape is %s. Square matrix expected." % P.shape)

    # убеждаемся, что сумма в каждой строке равна единице
    sums = np.sum(P, axis=1)
    if not np.allclose(sums, np.ones(P.shape[0])):
        raise ValueError("Matrix rows are not normalized. Row bindexToCount: %s." % sums)

    return P


class AbsorbingMarkovChain:

    # здесь я вычислю компоненты матрицы, описанные в статье https://en.wikipedia.org/wiki/Absorbing_Markov_chain
    # Из них легко получить много интересных вероятностей. Но здесь реализованы лишь необходимые

    def __init__(self, P: ndarray):
        ensure_stochastic_matrix(P)

        # я хочу привести матрицу P в "каноническую форму". Она по-прежнему останется корректной
        # стохастической матрицей, но состояния будут переупорядочены.
        #
        # В разных источниках канонической называют разные формы:
        #
        # (Q R) или (I 0)      https://en.wikipedia.org/wiki/Absorbing_Markov_chain
        # (0 I)     (R Q)      https://math.dartmouth.edu/~doyle/docs/walks/walks.pdf
        #
        # I - это фрагмент большой матрицы, сам по себе являющийся единичной матрицей. Если поглощающих
        # состояний n, то I - единичная матрица размером [n,n]. Все поглощающие состояния собираются в этом
        # фрагменте (у них вероятность перейти в самих себя равна единице, и поэтому у них всегда была единица
        # на главной диагонали). Чтобы форма матрицы P стала канонической, достаточно переместить поглощающие
        # состояния в какой-то из углов. Фрагменты R,Q,0 создания не требуют: они образуются сами собой в
        # процессе перемещения.
        #
        # Какую из двух канонических форм я предпочту - в данном случае неважно. Все последующие вычисления
        # будут производиться над матрицами R и Q, а различия между формами останутся лишь на уровне "какими
        # индексами мы обозначили поглощающие состояния".
        #
        # Далее я выясняю, как я хочу переупорядочить состояния. Под словом "индекс" я понимаю одновременно
        # и индекс состояния, и индекс строки, и индекс колонки - т.к. все они равны.

        ones_indexes = np.where(P.diagonal() == 1.0)[0]  # индексы где на главной диагонали единица
        all_indexes = np.arange(len(P))  # просто все возможные индексы
        non_ones_indexes = np.setdiff1d(all_indexes, ones_indexes)  # индексы, где на главной не единица

        if len(ones_indexes) <= 0:
            raise ValueError("No absorbing states found.")

        # TODO Написать отдельные легко тестируемые функции для работы с индексами.
        # Чтобы легче было разбираться, что значат состояния в канонической матрице,
        # а также к какому исходному состоянию относится transient state i.

        # переупорядоченные индексы: в таком порядке я хочу видеть состояния-строки-колонки.
        # В дальнейшем прочитав значение sourceIndexes[canonicalIndex] можно будет узнать, какой иначально
        # был индекс у состояния, которое в канонической форме получило имя canonicalIndex
        self.source_indexes = np.append(ones_indexes, non_ones_indexes)

        # Переставляю строки+колонки, чтобы получить каноническую матрицу M.
        # Возможно, это можно сделать в одно действие, но я пока умею.
        m = P[self.source_indexes]  # переупорядочил строки
        m = m[:, self.source_indexes]  # переупорядочил колонки
        self.M = m

        # для спокойствия убедимся, что матрица осталась квадратной и нормализованной
        ensure_stochastic_matrix(self.M)

        # получили матрицу в "канонической форме"
        # 	M = (I 0)
        #     	(R Q)
        # I - единичная квадратная матрица размером len(ones_indexes)
        # R - вероятности переходов к "поглощающим состояниям"
        # Q - вероятности переходов к остальным состояниям

        n = len(ones_indexes)

        self.R = self.M[n:, :n]  # первые n столбцов от всех строк, кроме первых n строк
        self.Q = self.M[n:, n:]  # следующие столбцы от всех строк, кроме первых n строк

        self.absorbingStatesCount = n

        self._It = np.identity(self.Q.shape[0])  # единичная матрица размером [t,t]

        # матрица N называется фундаментальной. Элемент N[i,j] - это матожидание количества визитов
        # в состояние j, если мы начали с состояния i (и бродим, пока не попадём в в absorbing state)
        # [https://math.dartmouth.edu/archive/m20x06/public_html/Lecture14.pdf]
        self._N = np.linalg.inv(self._It - self.Q)
        self.visitsExpected = self._N  # для единообразности именования (visitsVariance, stepsExpected, ...)

        # canonicalIndexes[sourceIndex] позволит узнать индекс исходного состояния sourceIndex
        # в новой канонической матрице
        self.canonicalIndexes = [np.where(self.source_indexes == i)[0][0]
                                 for i in range(self.source_indexes.size)]

    @cached_property
    def steps_expected(self):
        # возвращает массив R, значение R[ti] которого значит, что из состояния ti в среднем нужно будет сделать
        # R[ti] шагов, прежде чем окажешься поглощённым _любой_ из ловушек.
        #
        # Индекс ti значит "transient state № ti" в канонической матрице. У меня пока не было хороших примеров,
        # чтобы организовать удобный пересчёт индексов и тестировать его как следует.

        # вне зависимости от количества поглощающих состояний, получается вертикальный вектор
        # вроде [[10],[4],[6]]. То есть, поглощающие состояния никак не различаются. Я верну его
        # в более плоском виде: [10,4,6].

        return self._t.flatten()

    @cached_property
    def absorbing_prob(self):
        return np.matmul(self._N, self.R)

    @cached_property
    def transient_prob(self):
        return np.matmul(self._N - self._It, np.linalg.inv(self._Ndg))

    @cached_property
    def _Ndg(self) -> np.ndarray:
        # diagonal matrix with the same diagonal as _N
        # домножаю матрицу _N на единичную диагональную при помощи hadamard product
        return np.multiply(np.identity(self._N.shape[0]), self._N)

    @cached_property
    def _t(self) -> np.ndarray:
        # The expected number of steps before being absorbed when starting
        # in transient state i is the ith entry of the vector
        v = np.ones((len(self._N), 1))
        return np.matmul(self._N, v)

    @cached_property
    def visits_variance(self):
        # The variance on the number of visits to a transient state j with starting at a transient state i
        # (before being absorbed) is the (i,j)-entry of the matrix N2

        # в википедии Nsq определяется как произведение Адамара, где матрица умножается на саму себя.
        # А я просто возведу в квадрат каждую ячейку.
        Nsq = np.square(self._N)

        N2 = np.matmul(self._N, 2 * self._Ndg - self._It) - Nsq

        return N2

    @cached_property
    def stepsVariance(self):
        # The variance on the number of steps before being absorbed when starting in transient state i is the
        # i-th entry of the vector

        vertical = np.matmul(2 * self._N - self._It, self._t) - np.square(self._t)
        assert vertical.shape[1] == 1
        return vertical.flatten()


class TestAMC(unittest.TestCase):

    # !!! некоторые методы, например, stepsNumVariance, visitsNumVariance, transientProbs
    # тестируются вместе с классом MarkovWalk (в другом файле), поскольку только там мне легко
    # вычислить соответствующие величины эмпирически

    def test_stackexchange(self):
        P = [[1.0, 0, 0, 0, 0, 0, 0, 0],
             [0.7, 0, 0.3, 0, 0, 0, 0, 0],
             [0, 0.5, 0, 0.5, 0, 0, 0, 0],
             [0, 0, 0.3, 0, 0.7, 0, 0, 0],
             [0, 0, 0, 0.6, 0, 0.4, 0, 0],
             [0, 0, 0, 0, 0.2, 0, 0.8, 0],
             [0, 0, 0, 0, 0, 0.1, 0, 0.9],
             [0, 0, 0, 0, 0, 0, 0, 1.0]]

        amc = AbsorbingMarkovChain(np.asarray(P))

        correctC = np.array([
            [1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0.],
            [0.7, 0., 0., 0.3, 0., 0., 0., 0.],
            [0., 0., 0.5, 0., 0.5, 0., 0., 0.],
            [0., 0., 0., 0.3, 0., 0.7, 0., 0.],
            [0., 0., 0., 0., 0.6, 0., 0.4, 0.],
            [0., 0., 0., 0., 0., 0.2, 0., 0.8],
            [0., 0.9, 0., 0., 0., 0., 0.1, 0.]])

        correctR = np.array([
            [0.7, 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.9]])

        correctQ = np.array([
            [0., 0.3, 0., 0., 0., 0.],
            [0.5, 0., 0.5, 0., 0., 0.],
            [0., 0.3, 0., 0.7, 0., 0.],
            [0., 0., 0.6, 0., 0.4, 0.],
            [0., 0., 0., 0.2, 0., 0.8],
            [0., 0., 0., 0., 0.1, 0.]
        ])

        self.assertTrue(np.array_equal(amc.M, correctC))
        self.assertTrue(np.array_equal(amc.R, correctR))
        self.assertTrue(np.array_equal(amc.Q, correctQ))
        self.assertTrue(np.array_equal(amc.source_indexes, [0, 7, 1, 2, 3, 4, 5, 6]))
        self.assertEqual(amc.canonicalIndexes, [0, 2, 3, 4, 5, 6, 7, 1])

        correctProbs = np.array([
            [0.88349515, 0.11650485],
            [0.61165049, 0.38834951],
            [0.33980583, 0.66019417],
            [0.22330097, 0.77669903],
            [0.04854369, 0.95145631],
            [0.00485437, 0.99514563]])
        self.assertTrue(np.allclose(amc.absorbing_prob, correctProbs))

        print("Stackexchange example tested.")

    def test_wikipedia(self):
        # https://en.wikipedia.org/wiki/Absorbing_Markov_chain

        # Consider the process of repeatedly flipping a fair coin until the sequence (heads, tails, heads) appears.
        # This process is modeled by an absorbing Markov chain with transition matrix

        P = [[1 / 2, 1 / 2, 0, 0],
             [0, 1 / 2, 1 / 2, 0],
             [1 / 2, 0, 0, 1 / 2],
             [0, 0, 0, 1]]

        amc = AbsorbingMarkovChain(np.asarray(P))

        # The first state represents the empty string, the second state the string "H", the third state the string
        # "HT", and the fourth state the string "HTH". Although in reality, the coin flips cease after the string
        # "HTH" is generated, the perspective of the absorbing Markov chain is that the process has transitioned
        # into the absorbing state representing the string "HTH" and, therefore, cannot leave.

        correct_q = np.array([
            [0.5, 0.5, 0.],
            [0., 0.5, 0.5],
            [0.5, 0., 0.]
        ])

        correct_n = np.array([
            [4., 4., 2.],
            [2., 4., 2.],
            [2., 2., 2.]
        ])

        self.assertTrue(np.array_equal(amc.Q, correct_q))
        self.assertTrue(np.array_equal(amc._N, correct_n))

        # собственно, пример википедии затеян для вычисления этого значения: ожидаемого кол-ва шагов
        ens = amc.steps_expected
        correctENS = np.array([10., 8., 6.])
        self.assertTrue(np.array_equal(ens, correctENS))

        correctNdg = np.array([
            [4., 0., 0.],
            [0., 4., 0.],
            [0., 0., 2.]])

        self.assertTrue(np.array_equal(amc._Ndg, correctNdg))

        print("Wikipedia example tested.")


if __name__ == "__main__":
    unittest.main()
