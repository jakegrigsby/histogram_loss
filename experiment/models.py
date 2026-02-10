"""Keras implementations of models

Includes HL Gaussian, HL OneBin, and Regression models.
HL-Gaussian and Regression models accept multidimensional input and output.
"""

from tensorflow import keras
import tensorflow as tf
from experiment.transforms import *
from experiment.multidense import MultiDense


class Regression(keras.Model):
    """Model that performs regression using features from a given backbone model.

    If the base model outputs features with shape (batchsize, x1, ..., xn)
    and the out_shape is (s1, ..., sm), the model will have outputs with shape
    (batchsize, x1, ..., x(n-1), s1, ..., sm).

    Params:
        base - the backbone model to learn features
        out_shape - the dimensions added to the base features
    """

    def __init__(self, base, out_shape=(), name="regression"):
        super().__init__(name=name)
        self.base = base
        self.reg = MultiDense(out_shape, individual=False)

    def call(self, inputs, training=None):
        """Perform regression on the features outputted by the base model.

        Params:
            inputs - the tensor to give to the base model

        Returns:
            a tensor with consisting of the regression predictions
            for the given inputs (shape described in class docstring)
        """
        features = self.base(inputs, training=training)
        return self.reg(features)


class HistModel(keras.Model):
    """Model that performs regression on features from a given backbone model
    using a histogram loss.

    If the base model outputs features with shape (batchsize, x1, ..., xn)
    and the out_shape is (s1, ..., sm), the model will have outputs with shape
    (batchsize, x1, ..., x(n-1), s1, ..., sm).

    Params:
        base - the backbone model to learn features
        centers - the centers of the histogram bins
            If the targets have shape (batchsize, y1, ..., yn), centers
            must be broadcastable to (n_bins, y1, ..., yn)
        transform - the histogram transform to apply to the targets
            to facilitate learning
        name - the name of the model
    """

    def __init__(self, base, centers, transform, name="HistModel", out_shape=()):
        super().__init__(name=name)
        self.base = base
        shape = out_shape + centers.shape[:1]
        self.dense = MultiDense(shape, individual=True)
        self.softmax = keras.layers.Softmax()
        self.transform = transform
        self.mean = HistMean(centers)
        self.hist_loss = keras.metrics.Mean("loss")

    def call(self, inputs, training=None):
        """Perform regression on the inputs using the histogram loss model.

        Params:
            inputs - the tensor to give to the base model
            training - flag indicating whether the model is called during training

        Returns:
            a tensor with consisting of the regression predictions
            for the given inputs (shape described in class docstring)
        """
        hist = self.get_hist(inputs, training=training)
        return self.mean(hist)

    def get_hist(self, inputs, training=None):
        """Obtain the binned probability vectors for the given inputs.

        Params:
            inputs - the tensor to give to the base model
            training - flag indicating whether the model is called during training

        Returns:
            a tensor containing the probabilities of each output falling in a given bin
        """
        features = self.base(inputs, training=training)
        hist = self.dense(features)
        hist = self.softmax(hist)
        return hist

    def train_step(self, data):
        """Update the model weights and metrics based on a single batch of data.

        Params:
            data - a batch of data in the form (x, y)
                typically a tf.data.Dataset where elements are a tuple of tensors

        Returns: a dict containing the metric values computed on data
        """
        x, y = data
        y_transformed = self.transform(y)

        with tf.GradientTape() as tape:
            hist = self.get_hist(x, training=True)
            loss = keras.losses.categorical_crossentropy(y_transformed, hist)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        y_pred = self.mean(hist)
        self.compiled_metrics.update_state(y, y_pred)
        self.hist_loss.update_state(loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Evaluate the data on a validation batch and compute the loss.

        Params:
            data - a batch of data in the form (x, y)
                typically a tf.data.Dataset where the elements are a tuple of tensors
        """
        x, y = data

        y_transformed = self.transform(y)
        hist = self.get_hist(x, training=False)

        loss = keras.losses.categorical_crossentropy(y_transformed, hist)
        self.hist_loss.update_state(loss)

        y_pred = self.mean(hist)
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


class HLGaussian(HistModel):
    """Keras model using a histogram loss with a truncated Gaussian
    distribution on the targets.

    Params:
        base - the backbone model used to learn features
        borders - the borders of the histogram bins
        sigma - the sigma parameter of the truncated Gaussian distribution

    If the inputs have shape (batchsize, x1, ..., xd), then
        borders should be broadcastable with (n_bins + 1, x1, ..., xd)
        and sigma should be broadcastable with (x1, ..., xd)
    """

    def __init__(self, base, borders, sigma, **kwargs):
        centers = (borders[:-1] + borders[1:]) / 2
        transform = TruncGaussHistTransform(borders, sigma)
        super().__init__(base, centers, transform, "HL-Gaussian")


class HLOneBin(HistModel):
    """Keras model using a histogram loss with one-hot encoding of the targets.

    Params:
        base - the backbone model used to learn features
        borders - the borders of the histogram bins
        dropout - the dropout parameter for the histogram layer
    """

    def __init__(self, base, borders):
        centers = (borders[:-1] + borders[1:]) / 2
        transform = OneHotTransform(borders)
        super().__init__(base, centers, transform, "HL-OneBin")


class HLUniform(HistModel):
    """Keras model using a histogram loss with a mixture of a Dirac delta
    distribution with uniform noise on the targets.

    Params:
        base - the backbone model used to learn features
        borders - the borders of the histogram bins
        dropout - the dropout parameter for the histogram layer
        eps - the epsilon parameter for the uniform noise on the targets
    """

    def __init__(self, base, borders, eps):
        centers = (borders[:-1] + borders[1:]) / 2
        transform = UniformTransform(borders, eps)
        super().__init__(base, centers, transform, "HL-Uniform")


class HLProjected(HistModel):
    """Keras model using a histogram loss with a projection onto the two nearest bins.

    Params:
        base - the backbone model used to learn features
        borders - the borders of the histogram bins
        dropout - the dropout parameter for the histogram layer
    """

    def __init__(self, base, borders):
        centers = (borders[:-1] + borders[1:]) / 2
        transform = ProjTransform(centers)
        super().__init__(base, centers, transform, "HL-Projected")


class HLMCGaussian(HistModel):
    """Keras model using a histogram loss with a *mean-corrected* truncated
    Gaussian distribution on the targets.

    Unlike standard HL-Gaussian, the Gaussian centre is iteratively shifted
    so that E_p[c] = target, removing the truncation-induced mean bias that
    degrades performance at high sigma.

    Params:
        base    - the backbone model used to learn features
        borders - the borders of the histogram bins
        sigma   - the sigma parameter of the Gaussian
    """

    def __init__(self, base, borders, sigma, **kwargs):
        centers = (borders[:-1] + borders[1:]) / 2
        transform = MeanCorrectedGaussianTransform(borders, sigma)
        super().__init__(base, centers, transform, "HL-MCGaussian")


class HLGibbs(HistModel):
    """Keras model using a histogram loss with maximum-entropy (Gibbs) targets.

    Instead of a truncated Gaussian, the target distribution is the
    maximum-entropy distribution over the bins whose mean equals the target
    value. This is a Gibbs distribution p_i ∝ exp(λ * c_i).

    Params:
        base - the backbone model used to learn features
        borders - the borders of the histogram bins
        n_iter - number of Newton-Raphson iterations for solving λ (default 10)
    """

    def __init__(self, base, borders, n_iter=10, **kwargs):
        centers = (borders[:-1] + borders[1:]) / 2
        transform = GibbsTransform(centers, n_iter=n_iter)
        super().__init__(base, centers, transform, "HL-Gibbs")


class HLMaxEnt(HistModel):
    """Keras model using a two-moment maximum-entropy histogram loss.

    The target distribution p_i ∝ exp(λ₁ c_i + λ₂ c_i²) is the maximum-
    entropy distribution over bins whose mean matches the target value and
    whose variance equals σ². This is the principled discrete analogue of
    HL-Gaussian: in the continuous limit, mean + variance constraints yield
    a Gaussian, but the discrete version is provably optimal (highest entropy)
    for any finite number of bins.

    Params:
        base - the backbone model used to learn features
        borders - the borders of the histogram bins
        sigma - standard deviation of the target distribution
        n_iter - number of Newton-Raphson iterations (default 10)
    """

    def __init__(self, base, borders, sigma, n_iter=10, **kwargs):
        centers = (borders[:-1] + borders[1:]) / 2
        transform = TwoMomentMaxEntTransform(centers, sigma, n_iter=n_iter)
        super().__init__(base, centers, transform, "HL-MaxEnt")


class HLCauchy(HistModel):
    """Keras model using a histogram loss with Student-t (heavy-tailed) targets.

    The target distribution uses a Student-t density evaluated at bin centers:
        y_i ∝ (1 + (1/ν)·((c_i - target)/γ)²)^(-(ν+1)/2)

    This provides peaked targets (strong gradient signal near the target) with
    heavy polynomial tails (full bin utilization). The tail parameter ν
    controls the interpolation: ν=1 is Cauchy, ν→∞ recovers Gaussian.

    Params:
        base - the backbone model used to learn features
        borders - the borders of the histogram bins
        gamma - scale parameter (analogous to σ in HL-Gaussian)
        nu - degrees of freedom (1 = Cauchy, larger → Gaussian)
    """

    def __init__(self, base, borders, gamma, nu=1.0, **kwargs):
        centers = (borders[:-1] + borders[1:]) / 2
        transform = StudentTTransform(centers, gamma, nu=nu)
        super().__init__(base, centers, transform, "HL-Cauchy")
