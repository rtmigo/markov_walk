# SPDX-FileCopyrightText: (c) 2018 Art Galkin <ortemeo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

# If you've ever wondered what comments look like in Russian, scroll down.

import multiprocessing
import random
import sys
import unittest
from functools import cached_property  # needs python 3.8
from typing import *

import numpy as np

from .absorbing_markov_chain import AbsorbingMarkovChain


def random_walk_stochastic_matrix(step_right_probs: Sequence[float]) -> Sequence[Sequence[float]]:
    # Стохастическая матрица:
    # * строки обозначают "сейчас", текущее состояние
    # * колонки обозначают "будущее", следующее расстояние
    #
    # Образно говоря, движение идёт по диагоналям вправо-вверх.
    #
    # P[i,j] - значение в строке i, колонке j.
    # P[i,j] - условная вероятность того, что если сейчас состояние i, следующим будет состояние j.
    # P[i,j] = P( X[t+1]=j | X[t]=i )
    #
    # Сумма значений в каждой строке равна 1.

    dim = len(step_right_probs) + 2

    P = [[0.0] * dim for _ in range(dim)]  # создаём нулевую матрицу

    # вписываем единицы в левый-верхний и правый-нижний углы
    P[0][0] = 1.0
    P[-1][-1] = 1.0

    # добавляем вероятности переходов
    for i, pRight in enumerate(step_right_probs):
        if not 0 <= pRight <= 1:
            raise ValueError("Probability value out of range: %f (index %d)." % (pRight, i))
        P[i + 1][i + 2] = pRight
        P[i + 1][i] = 1 - pRight

    return P


class TestRandomWalkStochasticMatrix(unittest.TestCase):
    def test(self):
        probs = [0.3, 0.5, 0.7, 0.4, 0.8, 0.9]

        P = random_walk_stochastic_matrix(probs)

        # убедимся, что в каждой строке сумма равна 1
        for row in P:
            self.assertAlmostEqual(sum(row), 1, 5)

        # и вообще что значения ожидаемые
        correct_p = [
            [1.0, 0, 0, 0, 0, 0, 0, 0],
            [0.7, 0, 0.3, 0, 0, 0, 0, 0],
            [0, 0.5, 0, 0.5, 0, 0, 0, 0],
            [0, 0, 0.3, 0, 0.7, 0, 0, 0],
            [0, 0, 0, 0.6, 0, 0.4, 0, 0],
            [0, 0, 0, 0, 0.2, 0, 0.8, 0],
            [0, 0, 0, 0, 0, 0.1, 0, 0.9],
            [0, 0, 0, 0, 0, 0, 0, 1.0]
        ]

        self.assertTrue(np.allclose(P, correct_p))


class MarkovWalk:

    def __init__(self, step_right_probs: Sequence[float]):
        self.P = random_walk_stochastic_matrix(step_right_probs)
        self.amc = AbsorbingMarkovChain(np.array(self.P))

    @cached_property
    def right_edge_probs(self) -> List[float]:
        """`walk.right_edge_probs[pos]` is the probability for a starting point `pos`, that after infinite wandering we
        will leave the table on the right, and not on the left."""

        # совершаем случайное блуждание между точками.
        # В точке i вероятность шага вправо равна stepRightProbs[i].
        # В точке i вероятность шага влево равна 1-stepRightProbs[i].
        # Блуждание продолжается, пока мы не выйдем за пределы списка (слева или справа).
        #
        # Эта функция возвращает список, каждый элемент i которого равен вероятности выйти за пределы
        # списка именно справа, а не слева.

        # получилась матрица с двумя колонками. Строки соответствуют исходным состояниям.
        # В левой колонке вероятность пересечь левую границу, в правой - вероятность пересечь правую.
        # Поскольку сумма значений каждой строки равна 1, и колонок всего две, это несколько избыточно.
        # Я верну из функции только значения правой колонки.

        # return [row[1] for row in self.amc.absorbingProbs()]

        return self.amc.absorbing_prob[:, 1].flatten()

    @property
    def ever_reach_probs(self) -> np.ndarray:
        """Returns a matrix so that `ever_reach_probs[startPos][endPos]` is the probability, that after
        infinite wandering started at `startPos` we will ever reach the point `endPos`."""
        return self.amc.transient_prob


class TestWalk(unittest.TestCase):

    @staticmethod
    def walkToEdge(step_right_probs: Sequence[float], start_pos: int):
        visits = [0] * len(step_right_probs)
        pos = start_pos
        for step in range(1, sys.maxsize):
            if random.random() < step_right_probs[pos]:
                pos += 1
            else:
                pos -= 1
            if pos < 0:
                return -1, step, visits
            if pos >= len(step_right_probs):
                return +1, step, visits
            visits[pos] += 1

    @staticmethod
    def measureWalksToEdge(stepRightProbs: Sequence[float], startPos, n=1000000) -> Tuple[float, float, np.ndarray]:

        # совершаем случайное блуждание между точками.
        # В точке i вероятность шага вправо равна stepRightProbs[i].
        # В точке i вероятность шага влево равна 1-stepRightProbs[i].
        # Блуждание продолжается, пока мы не выйдем за пределы списка (слева или справа).
        # Эмпирически вычисляем вероятность выйти именно справа, а не слева.

        def yieldAB(a, b):
            for _ in range(n):
                yield a, b

        with multiprocessing.Pool() as pool:
            results = pool.starmap(TestWalk.walkToEdge, yieldAB(stepRightProbs, startPos))

        print(". ", end="")
        sys.stdout.flush()

        step_counts = list()
        counted_visits_vectors = list()

        # подсчитываем кол-во "выходов вправо"
        absorbed_right = 0
        for edge, stepsBeforeEdge, visits in results:
            if edge == 1:
                absorbed_right += 1
            step_counts.append(stepsBeforeEdge)
            counted_visits_vectors.append(visits)

        npa = np.array(counted_visits_vectors)

        pRight = absorbed_right / n  # вероятность выйти вправо

        # дисперсия количества визитов в каждое из состояний прежде, чем выпадем за границу
        visits_variances = np.var(npa, axis=0)

        # дисперсия общего количество шагов которое сделаем прежде, чем выпадем за границу
        step_counts_variance: float = float(np.var(step_counts))

        return pRight, step_counts_variance, visits_variances

    @staticmethod
    def measureWalksToPoint(step_right_probs: Sequence[float], start_pos, end_pos, n=1000000) -> float:

        # совершаем случайное блуждание между точками.
        # В точке i вероятность шага вправо равна stepRightProbs[i].
        # В точке i вероятность шага влево равна 1-stepRightProbs[i].
        # Блуждание продолжается, пока мы не выйдем за пределы списка (слева или справа).
        # Эмпирически вычисляем вероятность оказаться в точке endPos до того, как выйдем за пределы списка.

        import random

        def walk():
            pos = start_pos
            while True:
                if random.random() < step_right_probs[pos]:
                    pos += 1
                else:
                    pos -= 1
                if pos == end_pos:
                    return 1

                if pos < 0 or pos >= len(step_right_probs):
                    return 0

        # если попадаем в точку, получаем 1, а если не успеваем, то  0.
        # Просто просуммировав результат сразу получаем кол-во попаданий в точку.
        reached_times = sum(walk() for _ in range(n))

        # возвращаем вероятность попасть в эту точку раньше, чем в ловушку
        return reached_times / n

    def test_edge(self):

        step_right_probs = [0.3, 0.5, 0.7, 0.4, 0.8, 0.9]
        mv = MarkovWalk(step_right_probs)

        self.assertEqual(len(step_right_probs), len(mv.right_edge_probs))

        amc_probs = mv.right_edge_probs
        amc_step_num_vars = mv.amc.stepsVariance
        amc_visits_vars = mv.amc.visits_variance

        self.assertTrue(np.allclose(amc_probs,
                                    [0.11650485436893206, 0.38834951456310685, 0.6601941747572817, 0.7766990291262137,
                                     0.9514563106796117, 0.9951456310679612]))

        # сравним ожидаемые (вычисленные алгоритмом) вероятности с эмпирическими

        print("Edge: comparing theoretical to empirical probabilities:")
        n = 1250000
        for start_pos in range(len(step_right_probs)):
            print("  pos %d..." % start_pos, end="")
            sys.stdout.flush()

            empirical_p, empirical_var, empirical_visit_variances = TestWalk.measureWalksToEdge(step_right_probs,
                                                                                                start_pos, n)

            expected_p = amc_probs[start_pos]
            the_var = amc_step_num_vars[start_pos]

            print("P_em=%.5f P_th=%.5f V_em=%.5f V_th=%.5f" % (empirical_p, expected_p, empirical_var, the_var))
            self.assertAlmostEqual(empirical_p, expected_p, 2)
            self.assertAlmostEqual(empirical_var, the_var, 0)

            # а эти значения я не печатаю, поскольку их много. Нужно сравнить целую строку со строкой матрицы.
            # (и эмпирическое вычисление этой строки было самым долгим)
            self.assertTrue(np.allclose(amc_visits_vars[start_pos], empirical_visit_variances, atol=0.05))

        print("test_edge ok!")

    def test_transient(self):

        step_right_probs = [0.3, 0.5, 0.7, 0.4, 0.8, 0.9]
        tprobs = MarkovWalk(step_right_probs).ever_reach_probs  # walkToProbs(step_right_probs)

        self.assertEqual(len(step_right_probs), len(tprobs))

        # сравним ожидаемые (вычисленные алгоритмом) вероятности с эмпирическими

        print("Transient: comparing theoretical to empirical probabilities:")

        import random

        n = 2500000

        # составляем список точек из которых и в которые хотим попасть
        pairs_to_test = list()

        # случайные сочетания
        for _ in range(5):
            startPos = random.randint(0, len(step_right_probs) - 1)
            endPos = random.randint(0, len(step_right_probs) - 1)
            pairs_to_test.append((startPos, endPos))

        # и неслучайно совпадающие (где выходим из той же позиции, куда и идём)
        for _ in range(3):
            startPos = endPos = random.randint(0, len(step_right_probs) - 1)
            pairs_to_test.append((startPos, endPos))

        # перебираем запланированные точки и замеряем вероятности
        for startPos, endPos in pairs_to_test:
            print("  pos %d->%d... " % (startPos, endPos), end="")
            sys.stdout.flush()
            empirical_p = TestWalk.measureWalksToPoint(step_right_probs, startPos, endPos, n)
            expected_p = tprobs[startPos][endPos]
            print("empirical %.5f theoretical %.5f" % (empirical_p, expected_p))
            self.assertAlmostEqual(empirical_p, expected_p, 2)

        print("test_transient ok!")


if __name__ == "__main__":
    unittest.main()
