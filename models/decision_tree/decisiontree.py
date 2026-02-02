from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import json


@dataclass
class Node:
    """
    Узел дерева решений.

    Если узел является ЛИСТОМ:
      - feature is None, threshold is None
      - value хранит предсказание

    Если узел ВНУТРЕННИЙ:
      - feature и threshold определяют правило разбиения:
            X[:, feature] <= threshold -> left
            иначе -> right
    """
    feature: int = None
    threshold: float = None
    n_samples: int = None
    value: float = None
    mse: float = None
    left: Node = None
    right: Node = None

@dataclass
class DecisionTreeRegressor:
    """
    Дерево решений для регрессии (критерий MSE), построение depth-wise (в глубину).

    stop-criteria:
      - max_depth: ограничение глубины (None = без ограничения)
      - min_samples_split: минимальное число объектов для разбиения узла
    """
    max_depth: int | None
    min_samples_split: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """Обучение дерева на данных (X, y)."""
        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание для матрицы X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted values.
        """
        preds = [self._predict_one_sample(row) for row in X]
        return np.asarray(preds, dtype=float)

    def _predict_one_sample(self, features: np.ndarray) -> float:
        """
        Предсказание для одного объекта: спускаемся по дереву до листа
        и возвращаем node.value.
        """
        node = self.tree_

        while node.left is not None and node.right is not None:
            f = node.feature
            thr = node.threshold
            if f is None or thr is None:
                break
            if features[f] <= thr:
                node = node.left
            else:
                node = node.right
        return float(node.value)

    # ---------- критерии качества сплита ----------

    def _mse(self, y: np.ndarray) -> float:
        """
        MSE узла: средняя квадратичная ошибка относительно среднего значения,
        принимает на вход вектор значений, возвращает значение mse
        """
        mse_e = np.mean((y - np.mean(y)) ** 2)
        return float(mse_e)

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Взвешенный MSE сплита,
        принимает на вход два вектора значений (слева и справа) и возвращает средневзвешенную mse.
        """
        mse_w = ((self._mse(y_left) * y_left.size + self._mse(y_right) * y_right.size) / (y_left.size + y_right.size))
        return float(mse_w)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None]:
        """
        Поиск лучшего разбиения узла.
        Перебираем признаки и кандидаты порогов, выбираем минимальный weighted MSE.

        Псевдокод
        Перебираем все признаки:
            Перебираем все значения признака:
                Разделяем данные по (признак, порог) -> подвыборка слева, подвыборка справа
                    Считаем взвешенный критерий для подвыборок слева и справа
        Возвращаем индекс признака и значение порога, которые дают минимальное значение взвешенного критерия
        """
        y = np.asarray(y).ravel()
        best_score = np.inf
        best_feature: int | None = None
        best_threshold: float | None = None

        n_features = X.shape[1]
        for i in range(n_features):
            x = X[:, i]
            vals = np.unique(x)
            if vals.size < 2:
                continue

            for t in vals:
                left_mask = x <= t
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                score = self._weighted_mse(y[left_mask], y[right_mask])
                if score < best_score:
                    best_score = score
                    best_feature = i
                    best_threshold = float(t)

        return best_feature, best_threshold

    # ---------- рекурсивное построение дерева ----------
        
    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Рекурсивно строит поддерево для текущего (X, y).
        Базовый случай: достигнут критерий остановки -> лист.
        """
        y = np.asarray(y).ravel()
        n_samples = y.size
        node_mse = self._mse(y)
        node_value = float(np.mean(y))

        # корректная проверка глубины (None = без ограничения)
        depth_reached = (self.max_depth is not None) and (depth >= self.max_depth)

        # базовые случаи (лист)
        if depth_reached or n_samples < self.min_samples_split or node_mse == 0.0:
            return Node(
                feature=None,
                threshold=None,
                n_samples=int(n_samples),
                value=node_value,
                mse=node_mse,
                left=None,
                right=None
            )

        # пытаемся найти лучший сплит
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None or best_threshold is None:
            return Node(
                feature=None,
                threshold=None,
                n_samples=int(n_samples),
                value=node_value,
                mse=node_mse,
                left=None,
                right=None,
            )

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return Node(
                feature=None,
                threshold=None,
                n_samples=int(n_samples),
                value=node_value,
                mse=node_mse,
                left=None,
                right=None,
            )

        # рекурсивно строим детей
        left_child = self._split_node(X[left_mask], y[left_mask], depth + 1)
        right_child = self._split_node(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature=int(best_feature),
            threshold=float(best_threshold),
            n_samples=int(n_samples),
            value=node_value,
            mse=node_mse,
            left=left_child,
            right=right_child,
        )

    # ---------- экспорт дерева ----------

    def as_json(self) -> str:
        """Экспорт дерева в JSON"""
        return json.dumps(self._as_json(self.tree_), ensure_ascii=False)

    def _as_json(self, node: Node) -> dict | None:
        """Рекурсивно превращаем дерево в dict"""
        if node is None:
            return None

        # Лист: только value, n_samples, mse
        if node.left is None and node.right is None:
            return {
                "value": node.value if node.value is not None else None,
                "n_samples": int(node.n_samples) if node.n_samples is not None else 0,
                "mse": round(float(node.mse), 2) if node.mse is not None else 0.0,
            }

        # Внутренний узел
        return {
            "feature": int(node.feature) if node.feature is not None else None,
            "threshold": node.threshold if node.threshold is not None else None,
            "n_samples": int(node.n_samples) if node.n_samples is not None else 0,
            "mse": round(float(node.mse), 2) if node.mse is not None else 0.0,
            "left": self._as_json(node.left),
            "right": self._as_json(node.right),
        }

