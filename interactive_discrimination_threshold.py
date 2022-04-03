"""The module contains the class InteractiveDiscriminationThreshold
for building an interactive plot which helps to find the optimal
threshold dividing positive and negative classes in binary classifiers.
"""

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Dict, Tuple, Union

from app import build_plot


class InteractiveDiscriminationThreshold:
    """A class to build a discrimination threhold plot.

    ...

    Attributes
    ----------
    model: sklearn.base.ClassifierMixin
        Any binary classification model from scikit-learn (or scikit-learn
        pipeline with such a model as the last step) containing the method
        predict_proba() for predicting probability of the response.
    X: numpy.array
        2-dimensional training vector with predictors.
    y: numpy.array
        Response vectors relative to X.
    labels: numpy.array
        Binary vector where the first element (with index 0) is the label
        for the negative response value, and the second element (with
        index 1) corresponds to the positive response value.
    test_size: float
        Share of the test set (default is 0.2)
    input_df: pandas.DataFrame
        pandas dataframe with the generated values for threshold, precision,
        recall, F1 score, Q rate and other parameters to be used in the plot.

    Methods
    -------
    prepare_data(n_iter: int = 35) -> None:
        Iteratively generates columns for threshold, precision, recall,
        F1 score, Q rate and other parameters to be used in the plot and
        stores as the attribute input_df.
    _append_metrics(X: np.array, y_true: np.array) -> pd.DataFrame:
        Supplements prepare_data() with the values for prediction, recall,
        F1 score and queue rate per one iteration.
    _get_metrics(true_value: Union[str, int, float], predicted_prob: float,
    threshold: float) -> Tuple:
        Used as a helper of the internal method _append_metrics() for row-wise
        calculation of precision, recall, F1 score and queue rate.
    plot(plot_inline: bool = True, width: int = 1200, height: int = 550)
    -> None
        Generates an interactive version of the Discrimination Threshold
        plot.
    """
    
    def __init__(self,
                 model: ClassifierMixin,
                 X: np.array,
                 y: np.array,
                 label_dict: Dict[int, Union[int, float, str]],
                 test_size: float = 0.2):
        """Constructs and checks the values of all the necessary attributes for
        creating a class instance.

        Parameters
        ----------
            model: sklearn.base.ClassifierMixin
                Any binary classification model from scikit-learn (or scikit-
                learn pipeline with such a model as the last step) containing
                the method predict_proba() for predicting probability of the
                response.
            X: numpy.array:
                2-dimensional training vector with predictors.
            y: numpy.array
                Response vectors relative to X.
            labels: numpy.array
                Binary vector where the first element (with index 0) is the
                label for the negative response value, and the second element
                (with index 1) corresponds to the positive response value.
            test_size: float
                A float value between 0 and 1 corresponding to the share of
                the test set.

        Returns
        -------
            None
        """
        self.model = model
        self.X = X
        self.y = y
        self.labels = np.array(list(label_dict.values()))
        self.test_size = test_size
        self.input_df = pd.DataFrame()
        if (self.X.shape[0] != y.shape[0]):
            raise ValueError('Prediction and response variables have'
                             'different dimensions.')
        if set(self.labels) != set(np.unique(y)):
            raise ValueError("The values of 'label_dict' parameter don't "
                             "the values of 'y'.")
        if not 0 < self.test_size < 1:
            raise ValueError("The value of the parameter test_size must be "
                             "strictly larger that 0 and smaller than 1")
        try:
            self.model.fit(self.X, self.y)
        except ValueError:
            print('Make sure that the model is trained and the input data '
                  'is properly transformed.')

    def prepare_data(self, n_iter: int = 35, store_data: bool = False) -> None:
        """Iteratively generates columns for threshold, precision, recall,
        F1 score, Q rate and other parameters to be used in the plot and
        stores as the attribute input_df.

        Parameters
        ----------
            n_iter: int, optional
                The number of samples with the metrics to be generated
                (default is 35)
            store_data: bool, default
                If True, stores the generated input data into the file
                'data.csv' (default is False)
        Returns
        -------
            None
        """
        for train_iter in tqdm(range(1, n_iter + 1)):
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=self.test_size,
                random_state=train_iter)
            self.model.fit(X_train, y_train)
            df_tmp = self._append_metrics(X_test, y_test)
            df_tmp['train_iter'] = train_iter
            self.input_df = pd.concat([self.input_df, df_tmp])
            if store_data:
                self.input_df.to_csv('data.csv')

    def _append_metrics(self, X: np.array, y_true: np.array) -> pd.DataFrame:
        """Supplements prepare_data() with the values for prediction, recall,
        F1 score and queue rate per one iteration.

        Parameters
        ----------
            X: numpy.array
                Subsamples of the test set predictor values to be used for
                the generation of the metrics.
            y_true: numpy.array
                Vector with response values for the generated test set

        Returns
        -------
            pandas DataFrame with the columns for all the metrics generated
            during one iteration.
        """
        predicted_prob = self.model.predict_proba(X)[:, 1]
        df = pd.DataFrame()
        df['thresholds'] = np.arange(0, 1.01, 0.02)
        df[['precision', 'recall', 'f1', 'queue_rate']] = pd.DataFrame(
            (self._get_metrics(y_true, predicted_prob, row.thresholds)
             for row in df.itertuples())
        )
        return df

    def _get_metrics(self, true_value: Union[str, int, float],
                     predicted_prob: float, threshold: float) -> Tuple:
        """Used as a helper of the internal method _append_metrics() for row-wise
        calculation of precision, recall, F1 score and queue rate.

        Parameters
        ----------
            true_value: Union[str, int, float]
                A true value for the response.
            predicted_prob: float
                Predicted probability value of the estimator.
            threshold: float
                The threshold for dividing response values into positive and
                negative classes.

        Returns
        -------
            A tuple with the metrics to be sent back to the method
            _append_metrics()
        """
        pred_label = self.labels[(predicted_prob >= threshold).astype(int)]
        tp = np.sum((pred_label == self.labels[1])
                    & (true_value == self.labels[1]))
        fp = np.sum((pred_label == self.labels[1])
                    & (true_value == self.labels[0]))
        fn = np.sum((pred_label == self.labels[0])
                    & (true_value == self.labels[1]))
        pr = tp / (tp + fp) if tp + fp != 0 else 1
        rec = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = 2 / (pr**(-1) + rec**(-1)) if pr * rec != 0 else 0
        queue_rate = np.mean(predicted_prob >= threshold)
        return (pr, rec, f1, queue_rate)

    def plot(self, width: int = 1200, height: int = 550,
             app_mode: str = 'inline') -> None:
        """Generates an interactive version of the discrimination threshold
        plot by calling the function build_plot from app.py

        Parameters
        ----------
            width: int, optional
                Width of the plot (default is 1200)
            height: int, optional
                Height of the plot (default is 550)
            app_mode: str, optional
                If 'inline', the app is being created as instance of
                JupyterDash and the plot is drawn inside the notebook; if
                'external', then the app is created as JupyterDash
                instance and the plot is drawn in separate window
                (default is 'inline')
        Returns
        -------
            None
        """
        app = build_plot(self.input_df, width, height, app_mode=app_mode)
        app.run_server(mode=app_mode)


