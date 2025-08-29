 Understanding entropy in machine learning

This project is a practical and visual explanation of the key concepts of **entropy**, **information gain** and **cross-entropy** in the context of machine learning. Suitable for beginners in Data Science and ML.

![Example of graphs](entropy_analysis.png)

---

## 📂 Project structure

- `entropy_tutorial.py ` is the main script with graphs and explanations
- `entropy_analysis.png' — visualization results
- `README.md ` is a guide

---

## 📈 Explanation of graphs

### 1. **Entropy and probability (two classes)**

![Chart 1](https://i.imgur.com/entropy1.png )

> **What shows:** How the entropy changes when the probability of one of the two classes changes.

- At `p = 0.5` — maximum uncertainty → **entropy = 1.0**
- At `p = 0.1` or `p = 0.9` — the model is almost certain → **entropy is close to 0**
- Formula:
\[
H(p) = -p \log_2 p - (1-p) \log_2 (1-p)
\]

💡 **Conclusion:** Entropy measures "chaos" or "uncertainty" in a distribution.

---

### 2. **Entropy in different classification scenarios**

![Graph 2](https://i.imgur.com/entropy2.png )

> **What shows:** Comparison of entropy for different levels of confidence of the model.

| Scenario | Entropy | Interpretation |
|-----------------------|---------|---------------|
| Very confident (0.9, 0.1) | ~0.469 | Low uncertainty |
| Not at all confident (0.5, 0.5) | 1.0 | Maximum uncertainty |
| Slightly more confident (0.6, 0.4) | ~0.971 | Almost like a "coin" |

💡 **Conclusion:** The closer the probabilities are to 0.5, the higher the entropy → the model is not sure.

---

### 3. **Division of classes based on**

![Graph 3](https://i.imgur.com/entropy3.png )

> **What shows:** Two classes with different distributions. We are looking for the threshold that best separates them.

- Class 0: normal distribution ~ N(2.1)
- Class 1: normal distribution ~N(5.1)
- **Red line** is the best separation threshold

💡 **Conclusion:** A good feature allows you to clearly separate classes.

---

### 4. **Information Gain (IG) for different thresholds**

![Chart 4](https://i.imgur.com/entropy4.png )

> **What shows:** How **Information Gain** changes depending on the threshold.

- **IG = Entropy before partitioning is the weighted entropy after**
- Maximum IG is the best candidate for partitioning in the decision tree
- In the example, the maximum is near the threshold **3.5** , IG ≈ 0.37

💡 **Conclusion:** Information Gain helps you choose **the best partitioning** in algorithms like ID3, C4.5, CART.

---

### 5. **Cross-entropy (loss function)**

![Chart 5](https://i.imgur.com/entropy5.png )

> **What shows:** Dependence of **cross-entropy** on predicted probabilities.

- True probability: `[0.8, 0.2]`
- Cross-entropy:
\[
H(p,q) = -p \log q - (1-p) \log(1-q)
\]
- Minimum at `q = 0.8` — when the model guesses the truth

💡 **Conclusion:** Cross-entropy is a standard loss function in binary and multiclass classification.

---

## 🧠 Key findings

✅ **Entropy** is a measure of uncertainty.  
✅ **Information Gain** — how much the attribute "organizes" the data.  
✅ **Cross-entropy** is a loss function that penalizes poor predictions.  
 The more confident the model is, the lower the entropy.  
 IG is maximized in decision trees, and cross-entropy is minimized in neural networks.

---

## 🚀 How to launch

``bash
#1. Clone the repository
git clone https://github.com/YOUR_USERNAME/entropy-in-machine-learning.git

#2. Install dependencies
pip install numpy matplotlib seaborn scipy

#3. Run
python entropy_tutorial.py
