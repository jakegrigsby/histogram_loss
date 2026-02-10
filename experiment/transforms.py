"""
Module containing histogram transforms for targets.
"""

import tensorflow as tf
from tensorflow import keras


class TruncGaussHistTransform(keras.layers.Layer):
    """Layer that transforms a scalar target into a binned probability vector
    that approximates a truncated Gaussian distribution with the target as the mean.

    Params:
        borders - the borders of the histogram bins
        sigma - the sigma parameter of the truncated Gaussian distribution

    If the inputs have shape (batchsize, x1, ..., xd), then
        borders should be broadcastable with (n_bins + 1, x1, ..., xd)
        and sigma should be broadcastable with (x1, ..., xd)

    e.g. borders can be produced using linspace(low, high, n_bins + 1)
    """

    def __init__(self, borders, sigma):
        super().__init__(trainable=False, name="TruncGaussHistTransform")
        self.borders = borders
        self.sigma = sigma
        k = len(self.borders.shape)
        self.perm_out = list(range(1, k + 1)) + [0]

    def call(self, inputs):
        """Transform the input and return it.

        Params:
            inputs - the tensor of targets to transform

        Returns:
            x_transformed - a tensor of shape (batchsize, x1, ..., xd, n_bins)
            consisting of the probability vectors for each target
        """

        border_targets = self.adjust_and_erf(
            tf.expand_dims(self.borders, 1), inputs, self.sigma
        )
        two_z = border_targets[-1] - border_targets[0]
        x_transformed = (border_targets[1:] - border_targets[:-1]) / two_z
        return tf.transpose(x_transformed, self.perm_out)

    def adjust_and_erf(self, a, mu, sig):
        """Calculate the erf of a after standardizing and dividing by sqrt(2)."""
        return tf.math.erf((a - mu) / (tf.math.sqrt(2.0) * sig))


class OneHotTransform(keras.layers.Layer):
    """Layer that transforms a target into a one-hot representation
    based on the histogram bin that it lies in.

    Params:
        borders - the borders of the histogram bins
    """

    def __init__(self, borders):
        super().__init__(trainable=False, name="OneHotTransform")
        self.borders = borders
        self.bin_size = borders[1] - borders[0]
        self.low = tf.reduce_min(borders)
        self.n_classes = tf.size(borders) - 1

    def call(self, inputs):
        """Transform the input and return it.

        Params:
            inputs - the tensor of targets to transform

        Returns:
            a tensor of shape (len(inputs), len(borders) - 1)
            consisting of the one-hot vectors for each target
        """
        adjusted = (inputs - self.low) / self.bin_size
        indices = tf.cast(adjusted, tf.int32)
        return tf.one_hot(indices, self.n_classes, dtype=tf.float32)


class UniformTransform(keras.layers.Layer):
    """Transform the target using a mixture of Dirac delta and uniform distributions.
    A target y is mapped to a binned probability vector with values epsilon if
    y is not in the bin and 1 - (k-1) * epsilon if y is in the bin.

    Params:
        borders - the borders of the histogram bins
        eps - the uniform noise parameter
    """

    def __init__(self, borders, eps):
        super().__init__(trainable=False, name="UniformTransform")
        self.onehot = OneHotTransform(borders)
        self.eps = eps
        k = tf.size(borders) - 1
        self.scale = 1 - k * self.eps

    def call(self, inputs):
        """Transform the input and return it.

        Params:
            inputs - the tensor of targets to transform

        Returns:
            a tensor of shape (len(inputs), len(borders) - 1) consisting of the
            binned probability vectors
        """
        onehot = self.onehot(inputs)
        return onehot * self.scale + self.eps


class ProjTransform(keras.layers.Layer):
    """Project the target uniformly onto the two nearest histogram bins.

    Params:
        centers - the bins centers
    """

    def __init__(self, centers):
        super().__init__(trainable=False, name="ProjTransform")
        self.w = centers[1] - centers[0]
        self.low = centers[0]
        self.centers = centers

    def call(self, inputs):
        """Return the binned probability vectors for the inputs.

        Params:
            inputs - the targets to transform

        Returns: a tensor of shape (len(inputs), len(centers)) containing the probability
            of the target falling in each bin
        """
        i = tf.cast(tf.math.floordiv(inputs - self.low, self.w), tf.int32)
        m = tf.gather_nd(self.centers, tf.expand_dims(i, 1))
        p = (inputs - m) / self.w
        n = tf.size(inputs)
        inds = tf.range(0, n)
        indices = tf.concat([tf.stack([inds, i], 1), tf.stack([inds, i + 1], 1)], 0)
        values = tf.concat([1 - p, p], 0)
        return tf.scatter_nd(indices, values, (n, tf.size(self.centers)))


class TwoMomentMaxEntTransform(keras.layers.Layer):
    """Layer that transforms a scalar target into the maximum-entropy distribution
    over histogram bins matching both the target mean and a specified variance.

    The target distribution is:
        p_i ∝ exp(λ₁ c_i + λ₂ c_i²)

    where (λ₁, λ₂) are Lagrange multipliers solved via 2D Newton-Raphson:
        E_p[c]  = target
        Var_p[c] = σ²

    The Jacobian is the 2×2 covariance matrix of the sufficient statistics
    [c, c²], which is inverted analytically at each step.

    Params:
        centers - the centers of the histogram bins, shape (n_bins,)
        sigma   - target standard deviation of the distribution
        n_iter  - number of Newton-Raphson iterations (default 10)
    """

    def __init__(self, centers, sigma, n_iter=10):
        super().__init__(trainable=False, name="TwoMomentMaxEntTransform")
        self.centers = tf.cast(centers, tf.float32)  # (N,)
        self.centers_sq = tf.square(self.centers)  # (N,)
        self.sigma = tf.cast(sigma, tf.float32)
        self.target_var = tf.square(self.sigma)
        self.n_iter = n_iter

    def call(self, inputs):
        """Transform scalar targets into two-moment max-entropy probability vectors.

        Params:
            inputs - tensor of targets, shape (batchsize,)

        Returns:
            probs - tensor of shape (batchsize, n_bins)
        """
        targets = tf.expand_dims(tf.cast(inputs, tf.float32), -1)  # (B, 1)
        centers = tf.expand_dims(self.centers, 0)  # (1, N)
        centers_sq = tf.expand_dims(self.centers_sq, 0)  # (1, N)

        # Clamp targets to be strictly inside the bin range
        eps = 1e-3
        min_c = tf.reduce_min(self.centers)
        max_c = tf.reduce_max(self.centers)
        targets = tf.clip_by_value(targets, min_c + eps, max_c - eps)

        # Target second moment: E[c²] = mean² + σ²
        target_second = tf.square(targets) + self.target_var  # (B, 1)

        # Initialize with the continuous Gaussian approximation:
        #   p(c) ∝ exp(-(c-μ)²/(2σ²)) = exp(μc/σ² - c²/(2σ²))
        # So λ₁ ≈ μ/σ², λ₂ ≈ -1/(2σ²)
        inv_var = 1.0 / tf.maximum(self.target_var, 1e-12)
        lam1 = targets * inv_var  # (B, 1)
        lam2 = tf.fill(tf.shape(targets), -0.5 * inv_var)  # (B, 1)

        for _ in range(self.n_iter):
            logits = lam1 * centers + lam2 * centers_sq  # (B, N)
            probs = tf.nn.softmax(logits, axis=-1)  # (B, N)

            # Moments under current distribution
            ec1 = tf.reduce_sum(probs * centers, axis=-1, keepdims=True)  # E[c]
            ec2 = tf.reduce_sum(probs * centers_sq, axis=-1, keepdims=True)  # E[c²]
            ec3 = tf.reduce_sum(
                probs * centers * centers_sq, axis=-1, keepdims=True
            )  # E[c³]
            ec4 = tf.reduce_sum(
                probs * tf.square(centers_sq), axis=-1, keepdims=True
            )  # E[c⁴]

            # Residual: [E[c] - target, E[c²] - target_second]
            r1 = ec1 - targets  # (B, 1)
            r2 = ec2 - target_second  # (B, 1)

            # Freeze samples that have already converged — the Jacobian
            # becomes singular at the solution for peaked distributions,
            # so continuing to iterate would produce garbage steps.
            active = tf.cast(tf.abs(r1) + tf.abs(r2) > 1e-5, tf.float32)  # (B, 1)

            # Jacobian = Cov([c, c²]) with Tikhonov regularization:
            j11 = ec2 - tf.square(ec1) + 1e-8
            j12 = ec3 - ec1 * ec2
            j22 = ec4 - tf.square(ec2) + 1e-8

            # Analytical 2×2 inverse: [d -b; -b a] / det
            det = j11 * j22 - tf.square(j12)
            det = tf.maximum(det, 1e-16)

            step1 = (j22 * r1 - j12 * r2) / det
            step2 = (-j12 * r1 + j11 * r2) / det

            # Clip steps and only update unconverged samples
            step1 = tf.clip_by_value(step1, -500.0, 500.0) * active
            step2 = tf.clip_by_value(step2, -500.0, 500.0) * active

            lam1 = lam1 - step1
            lam2 = lam2 - step2

        # Final probabilities
        final_logits = lam1 * centers + lam2 * centers_sq
        return tf.nn.softmax(final_logits, axis=-1)


class GibbsTransform(keras.layers.Layer):
    """Layer that transforms a scalar target into a maximum-entropy (Gibbs)
    distribution over histogram bins whose mean matches the target.

    For a target value v and bin centers c_1, ..., c_N, we find the scalar
    Lagrange multiplier λ such that:
        p_i(λ) = softmax(λ * c_i)  and  Σ_i p_i * c_i = v

    λ is found via Newton-Raphson:  λ ← λ - (E[c] - v) / Var[c]

    Params:
        centers - the centers of the histogram bins, shape (n_bins,)
        n_iter  - number of Newton-Raphson iterations (default 10)
    """

    def __init__(self, centers, n_iter=10):
        super().__init__(trainable=False, name="GibbsTransform")
        self.centers = tf.cast(centers, tf.float32)  # (n_bins,)
        self.n_iter = n_iter

    def call(self, inputs):
        """Transform scalar targets into Gibbs probability vectors.

        Params:
            inputs - tensor of targets, shape (batchsize,)

        Returns:
            probs - tensor of shape (batchsize, n_bins)
        """
        targets = tf.expand_dims(tf.cast(inputs, tf.float32), -1)  # (B, 1)
        centers = tf.expand_dims(self.centers, 0)  # (1, N)

        # Clamp targets to be strictly inside the bin range
        eps = 1e-3
        min_c = tf.reduce_min(self.centers)
        max_c = tf.reduce_max(self.centers)
        targets = tf.clip_by_value(targets, min_c + eps, max_c - eps)

        # Newton-Raphson: start from λ=0 (uniform distribution)
        lambdas = tf.zeros_like(targets)  # (B, 1)

        for _ in range(self.n_iter):
            logits = lambdas * centers  # (B, N)
            probs = tf.nn.softmax(logits, axis=-1)  # (B, N)
            mean = tf.reduce_sum(probs * centers, axis=-1, keepdims=True)  # (B, 1)
            var = tf.reduce_sum(
                probs * tf.square(centers - mean), axis=-1, keepdims=True
            )  # (B, 1)
            step = (mean - targets) / tf.maximum(var, 1e-6)
            lambdas = lambdas - tf.clip_by_value(step, -10.0, 10.0)

        # Final probabilities
        final_logits = lambdas * centers
        return tf.nn.softmax(final_logits, axis=-1)


class StudentTTransform(keras.layers.Layer):
    """Layer that transforms a scalar target into a Student-t distribution
    over histogram bins, centered at the target value.

    Unlike HL-Gaussian (exponential tails → dead bins) or Gibbs (too diffuse),
    Student-t provides BOTH a peaked center AND heavy polynomial tails that
    keep every bin alive:

        y_i ∝ (1 + (1/ν)·((c_i - target)/γ)²)^(-(ν+1)/2)

    The tail parameter ν controls the interpolation:
        ν = 1  : Cauchy — heaviest tails (∝ 1/distance²), maximum full support
        ν = 5  : moderate tails
        ν → ∞  : recovers Gaussian (HL-Gauss)

    No Newton-Raphson needed — just evaluate the density at bin centers.

    Params:
        centers - the centers of the histogram bins, shape (n_bins,)
        gamma   - scale parameter (analogous to σ in HL-Gaussian)
        nu      - degrees of freedom (1 = Cauchy, larger → Gaussian)
    """

    def __init__(self, centers, gamma, nu=1.0):
        super().__init__(trainable=False, name="StudentTTransform")
        self.centers = tf.cast(centers, tf.float32)
        self.gamma = tf.cast(gamma, tf.float32)
        self.nu = tf.cast(nu, tf.float32)

    def call(self, inputs):
        """Transform scalar targets into Student-t probability vectors.

        Params:
            inputs - tensor of targets, shape (batchsize,)

        Returns:
            probs - tensor of shape (batchsize, n_bins)
        """
        targets = tf.expand_dims(tf.cast(inputs, tf.float32), -1)  # (B, 1)
        centers = tf.expand_dims(self.centers, 0)  # (1, N)

        z = (centers - targets) / self.gamma  # (B, N)
        log_probs = (
            -(self.nu + 1.0) / 2.0 * tf.math.log1p(tf.square(z) / self.nu)
        )  # (B, N)
        return tf.nn.softmax(log_probs, axis=-1)


class MeanCorrectedGaussianTransform(keras.layers.Layer):
    """Like TruncGaussHistTransform but iteratively shifts the Gaussian
    center so that E_p[c] = target (mean-preserving).

    Standard HL-Gaussian centres the Gaussian at the target value y, but
    truncation to the finite bin range biases E_p[c] toward the range
    centre, especially at high sigma.  This layer finds y' such that the
    truncated-Gaussian target centred at y' has E_p[c] = y via fixed-point
    iteration:  y' ← y' + (y − E_p[c | centre=y']).

    Params:
        borders - the borders of the histogram bins
        sigma   - standard deviation of the (untruncated) Gaussian
        n_iter  - maximum number of correction iterations (default 20)
    """

    def __init__(self, borders, sigma, n_iter=20):
        super().__init__(trainable=False, name="MeanCorrectedGaussianTransform")
        self.borders = tf.cast(borders, tf.float32)
        self.sigma = tf.cast(sigma, tf.float32)
        self.n_iter = n_iter
        centers = (borders[:-1] + borders[1:]) / 2
        self.centers = tf.cast(centers, tf.float32)  # (N,)

    def _gauss_probs(self, y_prime):
        """Compute truncated-Gaussian target probabilities centred at y_prime.

        Args:
            y_prime: (B, 1) centres

        Returns:
            probs: (B, N)
        """
        borders = tf.expand_dims(self.borders, 0)  # (1, N+1)
        scaled = (borders - y_prime) / (tf.math.sqrt(2.0) * self.sigma)
        cdf = tf.math.erf(scaled)  # (B, N+1)
        two_z = cdf[:, -1:] - cdf[:, :1]  # (B, 1)
        probs = (cdf[:, 1:] - cdf[:, :-1]) / (two_z + 1e-30)  # (B, N)
        return probs

    def call(self, inputs):
        """Transform scalar targets into mean-corrected Gaussian probability
        vectors.

        Params:
            inputs - tensor of targets, shape (batchsize,)

        Returns:
            probs - tensor of shape (batchsize, n_bins)
        """
        targets = tf.expand_dims(tf.cast(inputs, tf.float32), -1)  # (B, 1)
        centers = tf.expand_dims(self.centers, 0)  # (1, N)

        y_prime = tf.identity(targets)  # start with y' = y

        for _ in range(self.n_iter):
            probs = self._gauss_probs(y_prime)  # (B, N)
            mean = tf.reduce_sum(probs * centers, axis=-1, keepdims=True)  # (B, 1)
            residual = mean - targets  # (B, 1)
            # Stop iterating where converged
            active = tf.cast(tf.abs(residual) > 1e-7, tf.float32)
            y_prime = y_prime - residual * active

        return self._gauss_probs(y_prime)


class HistMean(keras.layers.Layer):
    """Layer that transforms a binned probability vector into its expected value.

    Params:
        centers - the centers of the histogram bins

    If inputs have shape (batchsize, x1, ..., xd, n_bins), then centers should be
    broadcastable to (n_bins, x1, ..., xd).
    """

    def __init__(self, centers):
        super().__init__(trainable=False, name="HistMean")
        k = len(centers.shape) - 1
        self.in_perm = list(range(1, k + 1)) + [0, k + 1]
        self.out_perm = [k] + list(range(k))
        centers_perm = list(range(1, k + 1)) + [0]
        self.centers = tf.transpose(centers, centers_perm)

    def call(self, inputs):
        """Return the weighted average between the bin centers and probability vectors.

        Params:
            inputs - a tensor of probability vectors to transform

        Returns:
            a tensor of shape (batchsize, x1, ..., xd) consisting of the expected values
        """
        inputs = tf.transpose(inputs, self.in_perm)
        means = tf.linalg.matvec(inputs, self.centers)
        return tf.transpose(means, self.out_perm)
