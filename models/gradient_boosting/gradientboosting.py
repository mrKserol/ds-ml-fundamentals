import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    def __init__(
            self,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=2,
            loss="mse",
            verbose=False,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.verbose = verbose

        self.base_pred_ = None
        self.trees_ = None

    def _mse(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        loss = np.mean((y_true - y_pred) ** 2)
        grad = (y_pred - y_true)

        return loss, grad

    def fit(self, X, y):
        """
        Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingRegressor: The fitted model.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.trees_ = []

        # Делаем первое предсказанием (средним значением)
        self.base_pred_ = y.mean()

        if self.loss == "mse" or self.loss is None:
            loss_fn = self._mse
        elif callable(self.loss):
            loss_fn = self.loss
        else:
            raise ValueError("loss must be 'mse' or a callable")

        predictions = np.full_like(y, self.base_pred_, dtype=float)

        # Вычислим вектор градиента функции потерь
        for _ in range(self.n_estimators):
            loss_value, grad = loss_fn(y, predictions)
            residuals = -grad
            tree = DecisionTreeRegressor(
                criterion="squared_error",
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=0,
            )

            # Обучим следующее дерево в ансамбле, где в качестве таргета используем вектор антиградиента
            tree.fit(X, residuals)

            # Сделаем шаг градиентного спуска (добавим к предсказаниям ансамбля предсказания нового дерева).
            # Чтобы шаг не получился слишком большим и мы случайно не перешагнули точку минимума, шаг умножим на
            # коэффициент скорости обучения (learning rate). Таким образом к предсказаниям ансамбля
            # добавляются новые y_pred * learning_rate

            predictions += self.learning_rate * tree.predict(X)

            # Сохраним дерево в список деревьев,
            # чтобы в будущем можно было формировать предсказания для новых объектов.
            self.trees_.append(tree)

            if self.verbose:
                print(loss_value)

        return self

    def predict(self, X):
        """Predict the target of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            y: array-like of shape (n_samples,)
            The predict values.

        """
        X = np.asarray(X)

        # Предсказываем средним значением, выученным моделью на обучающей выборке.
        predictions = np.full(X.shape[0], self.base_pred_, dtype=float)

        # Получаем предсказание от первого дерева в ансамбле.
        # Добавляем результат к итоговому предсказанию, умножив на learning rate.
        for tree in self.trees_:
            predictions += self.learning_rate * tree.predict(X)

        return predictions
