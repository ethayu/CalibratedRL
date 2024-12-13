# CalibratedRL: Calibrated Uncertainties for Model-Based Reinforcement Learning

This repository contains the official implementation for the paper **"Calibrated Uncertainties for Improved Model-Based Reinforcement Learning"**. The paper explores the importance of calibrated uncertainties in predictive models and introduces a simple yet effective method to improve performance using Isotonic Regression. 

---

## 📄 Paper Abstract

Estimates of predictive uncertainty are crucial for accurate model-based planning and reinforcement learning. However, predictive uncertainties — especially those derived from modern deep learning systems — are often inaccurate, limiting performance. 

This paper argues that good uncertainties must be calibrated, ensuring that predicted probabilities match empirical frequencies of events. We describe a straightforward approach to augment any model-based reinforcement learning agent with a calibrated model. Using **Isotonic Regression** for calibration, we demonstrate consistent improvements in:
- Planning
- Sample complexity
- Exploration