# Fraud Detection with Autoencoders (Unsupervised Learning)

This project explores how unsupervised learning — specifically autoencoders — can be used to detect fraudulent transactions without relying on labeled data.

In many real-world settings (like the one simulated here), fraud labels are often **delayed**, **missing**, or **incomplete**, which makes supervised models unreliable. The approach here assumes that fraud is rare, and trains the model to learn what "normal" behavior looks like — then flags deviations as suspicious.

I used the **IEEE-CIS Fraud Detection dataset** from Kaggle as a stand-in for a company's internal transaction stream. The column structure, size, and fraud rate are close to what many real payment systems experience, making it a good testbed.

---

## Project Scenario

> *"You're working for a fraud analytics team that wants to test autoencoders for fraud detection. Labels in production are often missing, so unsupervised methods are preferred. You're given a dataset that mimics the company's schema and asked to build a prototype pipeline — including baseline evaluation, feature engineering, and deployment to a test server."*

---

## What I Built

- **Baseline model** using only raw numeric features and a time-aware train/test split
- **Feature factory** with a config system to control feature sets
- **Feature engineering** for:
- Behavioral drift (rolling averages)
- Temporal patterns (hour, time gaps)
- Entity frequency/novelty (rarity of card/email/etc.)
- **Model selection** based on ROC AUC using W&B for tracking
- **Deployment** to a DigitalOcean droplet using FastAPI
- **Sample inference endpoint** that returns anomaly scores and fraud flags

---

## Metric of Interest
The primary evaluation metric is **ROC AUC**, which reflects how well the model separates normal vs. anomalous transactions.

---

## Stack
- Python, Pandas, Scikit-learn
- Pytorch (autoencoder)
- Weights & Biases for experiment tracking
- Docker + FastAPI for deployment

---

## Next Steps
I'm continuing to iterate on feature sets and thresholding logic, and plan to test ensemble models next.

---

> This project is a simulation — but it's built to reflect the types of problems real fraud detection teams face, including deployment readiness, model reproducibility, and production constraints.