# Machine Learning Optimizers Overview

---

## **1. Gradient Descent (Batch)**

* **Update Rule:**

  $$
  \theta \leftarrow \theta - \eta \nabla J(\theta)
  $$

* **Description:** Uses the entire training dataset to compute the gradient each iteration.

  This method computes the gradient of the cost function with respect to the parameters using the entire dataset at each step. It provides a stable and accurate direction toward the minimum but can be computationally expensive when datasets are large. It's well-suited for convex optimization problems where convergence guarantees are stronger.

* **Pros:**

  * Stable and converges for convex problems.
  * Deterministic updates.

* **Cons:**

  * Expensive for large datasets.
  * Slow convergence on ill-conditioned problems.

* **Use Case:** Small datasets, convex problems (e.g. logistic regression, SVM).

---

## **2. Stochastic Gradient Descent (SGD)**

* **Update Rule:**

  $$
  \theta \leftarrow \theta - \eta \nabla J(\theta; x_i, y_i)
  $$

* **Description:** Updates parameters using one random training example at a time.

  By updating parameters using a single sample at a time, SGD drastically reduces computation per iteration, making it highly scalable. However, the noise in individual updates introduces variance, which may cause the cost function to fluctuate rather than decrease smoothly. Despite this, it can escape shallow local minima, which can be beneficial in non-convex settings like deep learning.

* **Pros:**

  * Fast updates.
  * Works well with large datasets.

* **Cons:**

  * High variance in updates.
  * Requires a learning rate schedule.

* **Use Case:** Large datasets, deep learning, online learning.

---

## **3. Mini-batch Gradient Descent**

* **Update Rule:**

  $$
  \theta \leftarrow \theta - \frac{\eta}{m} \sum_{i \in B} \nabla J(\theta; x_i, y_i)
  $$

* **Description:** Uses a subset (mini-batch) of data for each update.

  Mini-batch GD balances between the stability of batch gradient descent and the efficiency of SGD. By using a batch of $m$ examples, it achieves more accurate gradient estimation than SGD while being faster than full-batch methods. It also makes use of GPU acceleration more effectively, making it the standard for modern deep learning.

* **Pros:**

  * Efficient use of hardware.
  * Reduces variance compared to SGD.

* **Cons:**

  * Batch size must be tuned.

* **Use Case:** Standard for training neural networks.

---

## **4. Momentum**

* **Update Rule:**

  $$
  v \leftarrow \mu v + \nabla J(\theta) \\
  \theta \leftarrow \theta - \eta v
  $$

* **Description:** Accumulates past gradients to accelerate updates.

  Momentum helps speed up convergence, especially in the presence of high curvature or noisy gradients. It does this by smoothing updates: each update is a combination of the current gradient and the previous update direction. This helps "build velocity" in relevant directions and suppress oscillations across irrelevant ones.

* **Pros:**

  * Faster convergence.
  * Dampens oscillations.

* **Cons:**

  * Adds momentum hyperparameter.

* **Use Case:** Deep learning, high-dimensional problems.

---

## **5. Nesterov Accelerated Gradient (NAG)**

* **Update Rule:**

  $$
  v \leftarrow \mu v + \nabla J(\theta - \eta \mu v) \\
  \theta \leftarrow \theta - \eta v
  $$

* **Description:** Similar to momentum but uses a look-ahead gradient.

  NAG improves upon momentum by first making a partial update and then computing the gradient at the "look-ahead" position. This results in more responsive and accurate updates, especially when nearing the optimum, as it prevents overshooting by peeking ahead before committing to an update.

* **Pros:**

  * Smoother and faster convergence.

* **Cons:**

  * Slightly more complex than momentum.

* **Use Case:** Deep learning, especially when momentum is already used.

---

## **6. Adagrad**

* **Update Rule:**

  $$
  G_t = \sum_{\tau=1}^t (\nabla J(\theta_\tau))^2 \\
  \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla J(\theta_t)
  $$

* **Description:** Adapts learning rate based on past gradient magnitude.

  Adagrad automatically adjusts the learning rate for each parameter individually, scaling smaller for frequently updated parameters and larger for infrequently updated ones. This makes it particularly effective for sparse features, as it naturally gives more attention to rarely seen data.

* **Pros:**

  * Good for sparse data.
  * No manual learning rate decay needed.

* **Cons:**

  * Learning rate may decay too much over time.

* **Use Case:** Sparse features, NLP, text models.

---

## **7. RMSProp**

* **Update Rule:**

  $$
  E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma)(\nabla J(\theta_t))^2 \\
  \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \odot \nabla J(\theta_t)
  $$

* **Description:** Fixes Adagrad's aggressive decay using an exponential moving average.

  RMSProp maintains a decaying average of past squared gradients, preventing the learning rate from shrinking too quickly as in Adagrad. It adapts quickly to the geometry of the problem and is particularly effective in handling non-stationary objectives, such as those encountered in recurrent models or reinforcement learning.

* **Pros:**

  * Works well in non-stationary settings.
  * More stable than Adagrad.

* **Cons:**

  * Sensitive to learning rate and decay hyperparameters.

* **Use Case:** RNNs, noisy or changing data distributions.

---

## **8. Adam (Adaptive Moment Estimation)**

* **Update Rule:**

  $$
  m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla J(\theta_t) \\
  v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla J(\theta_t))^2 \\
  \hat{m}_t = \frac{m_t}{1 - \beta_1^t},\quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
  \theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
  $$

* **Description:** Combines momentum and RMSProp with bias correction.

  Adam maintains an exponentially decaying average of both past gradients and their squares, along with bias correction terms to account for initialization. This allows Adam to perform well across a wide range of problems without much hyperparameter tuning, making it the de facto standard optimizer in modern deep learning.

* **Pros:**

  * Fast convergence.
  * Little tuning needed.

* **Cons:**

  * Can diverge in rare cases.
  * Higher memory usage.

* **Use Case:** Default optimizer in deep learning (CNNs, RNNs, Transformers).

---

## **9. L-BFGS (Limited-memory BFGS)**

* **Update Rule:**
  Uses an approximation of the inverse Hessian to perform:

  $$
  d_t = -H_t \nabla J(\theta_t),\quad \theta_{t+1} = \theta_t + \alpha d_t
  $$

* **Description:** A quasi-Newton method that uses gradient history to estimate curvature.

  Unlike first-order methods, L-BFGS uses both gradients and an approximation of second-order information (Hessian) to inform its updates. It avoids explicitly computing the Hessian, instead using a limited memory version suitable for problems with many parameters. It converges quickly on convex problems but is not well-suited for deep networks or very large datasets due to memory and computation constraints.

* **Pros:**

  * Fast convergence on convex problems.
  * Requires fewer iterations.

* **Cons:**

  * Expensive per iteration.
  * Less suited for large-scale or deep learning.

* **Use Case:** Convex optimization, logistic regression, small-scale ML.
