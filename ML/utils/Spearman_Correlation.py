import pandas as pd
import numpy as np
from scipy.stats import rankdata, t
import typing


class SpearmanCorrelation:
    def check_X(
        self, X: typing.Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> np.ndarray:
        """
        Check if X is pandas DataFrame, pandas Series or numpy array and convert it to numpy array.

        Args:
            X: (Union[pd.DataFrame, pd.Series, np.ndarray]): input data.

        Returns:
            X: (np.ndarray): converted input data.
        """
        if (
            not isinstance(X, pd.DataFrame)
            and not isinstance(X, pd.Series)
            and not isinstance(X, np.ndarray)
        ):
            raise TypeError(
                "Wrong type of X. It should be pandas DataFrame, pandas Series, numpy array."
            )
        X = np.array(X)
        if X.ndim == 1:
            X = X[None, :]
        return X

    def check_y(
        self, y: typing.Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> np.ndarray:
        """
        Check if y is pandas DataFrame, pandas Series or numpy array and convert it to numpy array.

        Args:
            y: (Union[pd.DataFrame, pd.Series, np.ndarray]): target data.

        Returns:
            y: (np.ndarray): converted target data.
        """
        if (
            not isinstance(y, pd.DataFrame)
            and not isinstance(y, pd.Series)
            and not isinstance(y, np.ndarray)
        ):
            raise TypeError(
                "Wrong type of y. It should be pandas DataFrame, pandas Series, numpy array."
            )
        y = np.array(y)
        if y.ndim != 1:
            y = y.squeeze()
        return y

    def check_for_object_columns(self, X: np.ndarray) -> np.ndarray:
        """
        Check if X contains object columns and convert it to numeric data.

        Args:
            X: (np.ndarray): input data.

        Returns:
            X: (np.ndarray): converted input data.
        """
        X = pd.DataFrame(X)
        if X.select_dtypes(include=np.number).shape[1] != X.shape[1]:
            raise TypeError(
                "Your data contains object or string columns. Numeric data is obligated."
            )
        return np.array(X)

    def fit(
        self,
        X: typing.Union[pd.DataFrame, pd.Series, np.ndarray],
        y: typing.Union[pd.DataFrame, pd.Series, np.ndarray],
        alpha: float = 0.05,
    ) -> None:
        """
        Perform Spearman correlation

        Args:
            X (Union[pd.DataFrame, pd.Series, np.ndarray]): input data.
            y (Union[pd.DataFrame, pd.Series, np.ndarray]): target data.
            alpha (float): significance level.
        """
        X = self.check_X(X)
        X = self.check_for_object_columns(X)
        y = self.check_y(y)
        ranked_X = rankdata(X)
        ranked_y = rankdata(y)
        covariance = self.calculate_covariance(X=ranked_X, y=ranked_y)
        stdX = self.calculate_std(data=ranked_X)
        stdY = self.calculate_std(data=ranked_y)
        self.correlation_ = self.calculate_correlation(
            covariance=covariance, stdX=stdX, stdY=stdY
        )
        self.test_statistic_ = self.calculate_test_statistic(
            correlation=self.correlation_, N=X.shape[0]
        )
        self.p_value_ = self.calculate_p_value_t_test(
            t_test=self.test_statistic_, df=X.shape[0] - 2
        )
        self.critical_value_ = self.calculate_critical_value(
            df=X.shape[0] - 2, alpha=alpha
        )
        self.keep_H0 = self.statistical_inference(p_value=self.p_value_, alpha=alpha)

    def calculate_covariance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate covariance between two arrays.

        Args:
            X (np.ndarray): input data.
            y (np.ndarray): target data.

        Returns:
            np.ndarray: covariance matrix.
        """
        return np.cov(X, y)

    def calculate_std(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate standard deviation of the data.

        Args:
            data (np.ndarray): input data.

        Returns:
            np.ndarray: standard deviation of the data.
        """
        return np.std(data)

    def calculate_correlation(
        self, covariance: np.ndarray, stdX: np.ndarray, stdY: np.ndarray
    ) -> np.ndarray:
        """
        Calculate correlation between two arrays.

        Args:
            covariance (np.ndarray): covariance matrix.
            stdX (np.ndarray): standard deviation of X.
            stdY (np.ndarray): standard deviation of Y.

        Returns:
            np.ndarray: correlation between X and Y.
        """
        return (covariance / (stdX * stdY))[1][0]

    def calculate_test_statistic(self, correlation: np.ndarray, N: int) -> np.ndarray:
        """
        Calculate test statistic.

        Args:
            correlation (np.ndarray): correlation between X and Y.
            N (int): number of samples.

        Returns:
            np.ndarray: test statistic.
        """
        return (correlation * np.sqrt(N - 2)) / np.sqrt(1 - correlation**2)

    def calculate_p_value_t_test(self, t_test: np.ndarray, df: int) -> np.ndarray:
        """
        Calculate p-value for t-test.

        Args:
            t_test (np.ndarray): test statistic.
            df (int): degrees of freedom.

        Returns:
            np.ndarray: p-value.
        """
        return 2 * (1 - t.cdf(np.abs(t_test), df))

    def calculate_critical_value(self, df: int, alpha: float) -> float:
        """
        Calculate critical value.

        Args:
            df (int): degrees of freedom.
            alpha (float): significance level.

        Returns:
            critical_value (float): critical value.
        """
        return t.isf(q=alpha / 2, df=df)

    def statistical_inference(self, p_value: float, alpha: float) -> bool:
        """
        Perform statistical inference.

        Args:
            p_value: (float): p value.
            alpha: (float): significance level.

        Returns:
            bool: (bool): True if H0 is not rejected, False otherwise.
        """
        if p_value >= alpha:
            return True
        else:
            return False
