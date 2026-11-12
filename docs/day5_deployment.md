# Day 5: ML Model Deployment & Collaboration

## Overview

Today we will learn how to deploy our ML model for **FREE** using GitHub and Streamlit Cloud. By the end of this session, your model will be accessible via a public URL that anyone can use.

---

## What We'll Cover

| Topic | Tool | Cost |
|-------|------|------|
| Code Hosting | GitHub | FREE |
| CI/CD Pipeline | GitHub Actions | FREE |
| Model Deployment | Streamlit Cloud | FREE |
| Interactive Dashboard | Streamlit | FREE |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MLOps Deployment Flow                         │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────┐         ┌──────────────────┐         ┌──────────────┐
    │  Local   │  push   │     GitHub       │  auto   │  Streamlit   │
    │   Code   │ ──────► │   Repository     │ ──────► │    Cloud     │
    └──────────┘         └──────────────────┘         └──────────────┘
                                  │
                                  │ triggers
                                  ▼
                         ┌──────────────────┐
                         │  GitHub Actions  │
                         │   (CI/CD)        │
                         │  - Lint code     │
                         │  - Run tests     │
                         │  - Build Docker  │
                         └──────────────────┘
```

---

## Part 1: Understanding CI/CD

### What is CI/CD?

| Term | Full Form | Purpose |
|------|-----------|---------|
| **CI** | Continuous Integration | Automatically test code on every push |
| **CD** | Continuous Deployment | Automatically deploy after tests pass |

### Our CI Pipeline (`.github/workflows/ci.yml`)

```yaml
name: CI Pipeline

on:
  push:
    branches: [master, develop]
  pull_request:
    branches: [master]

jobs:
  lint:
    name: Code Quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install flake8 black isort
      - run: black --check src/ tests/
      - run: isort --check-only src/ tests/

  test:
    name: Tests
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --cov=src
```

### CI Pipeline Jobs

| Job | What It Does | When It Fails |
|-----|--------------|---------------|
| **Code Quality** | Checks formatting (Black, isort) | Code not properly formatted |
| **Tests** | Runs pytest on all tests | Tests fail |
| **Pipeline Validation** | Runs ML pipeline end-to-end | Pipeline broken |
| **Build Docker** | Builds Docker image | Dockerfile error |

---

## Part 2: Code Formatting

### Why Code Formatting Matters

- Consistent code style across team
- Easier to review pull requests
- CI pipeline enforces standards

### Tools We Use

| Tool | Purpose | Command |
|------|---------|---------|
| **Black** | Code formatter | `black src/ tests/` |
| **isort** | Import sorter | `isort --profile black src/ tests/` |
| **flake8** | Linter (finds errors) | `flake8 src/ tests/` |

### Configuration (`pyproject.toml`)

```toml
[tool.black]
line-length = 88
target-version = ['py310']

[tool.isort]
profile = "black"
line_length = 88
```

### Running Locally

```bash
# Install tools
pip install black isort flake8

# Format code
black src/ tests/
isort --profile black src/ tests/

# Check for issues
flake8 src/ tests/
```

---

## Part 3: Streamlit App

### What is Streamlit?

- Python library for creating web apps
- No HTML/CSS/JavaScript needed
- Perfect for ML dashboards

### Our App (`app.py`)

```python
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Page config
st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

# Title
st.title("🌸 Iris Flower Classification")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Predict button
if st.button("Predict"):
    # Load model and predict
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.success(f"Predicted: {prediction}")
```

### Running Locally

```bash
# Install streamlit
pip install streamlit

# Run app
streamlit run app.py

# Opens browser at http://localhost:8501
```

---

## Part 4: Deploying to Streamlit Cloud

### Step 1: Push Code to GitHub

```bash
# Add all files
git add .

# Commit
git commit -m "Add Streamlit app"

# Push
git push origin master
```

### Step 2: Connect to Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"Sign in with GitHub"**
3. Authorize Streamlit

### Step 3: Deploy App

1. Click **"New app"**
2. Select your repository
3. Set:
   - **Branch:** `master`
   - **Main file:** `app.py`
4. Click **"Deploy"**

### Step 4: Access Your App

Your app will be available at:
```
https://[your-app-name].streamlit.app
```

**Our Live App:** https://d8vz9zebcwgfornj2e2ud3.streamlit.app

---

## Part 5: How Auto-Deploy Works

```
┌─────────────┐      ┌─────────────┐      ┌─────────────────┐
│ Make code   │      │   Push to   │      │ Streamlit Cloud │
│   changes   │ ───► │   GitHub    │ ───► │  auto-deploys   │
└─────────────┘      └─────────────┘      └─────────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │   GitHub    │
                     │   Actions   │
                     │  runs tests │
                     └─────────────┘
```

Every time you push:
1. GitHub Actions runs CI (tests, linting)
2. Streamlit Cloud detects changes
3. App automatically redeploys

---

## Part 6: Free Deployment Options

| Platform | Best For | Limitations |
|----------|----------|-------------|
| **Streamlit Cloud** | Dashboards, demos | 1GB memory |
| **Hugging Face Spaces** | Model demos | 2 vCPU, 16GB |
| **Render.com** | APIs | 750 hrs/month |
| **Railway.app** | Full apps | $5 credit/month |
| **GitHub Pages** | Static sites only | No Python backend |

---

## Part 7: Collaboration Features

### GitHub Features for Teams

| Feature | Purpose |
|---------|---------|
| **Issues** | Track bugs and features |
| **Pull Requests** | Review code before merging |
| **Projects** | Kanban boards for planning |
| **Discussions** | Team Q&A |
| **Actions** | Automated CI/CD |

### Sharing Your ML App

1. **Share URL** - Anyone can access the Streamlit app
2. **Fork Repo** - Others can copy and modify
3. **Collaborate** - Add team members to repo

---

## Summary

### What We Achieved Today

| Task | Status |
|------|--------|
| Fixed CI/CD pipeline | ✅ |
| Added code formatting config | ✅ |
| Created Streamlit app | ✅ |
| Deployed to Streamlit Cloud | ✅ |
| Made repo public | ✅ |

### Complete MLOps Stack (All FREE)

```
┌────────────────────────────────────────────────────────────┐
│                    Your MLOps Stack                        │
├────────────────────────────────────────────────────────────┤
│  Code         │  GitHub Repository                         │
│  CI/CD        │  GitHub Actions                            │
│  Tracking     │  MLflow (local)                            │
│  Deployment   │  Streamlit Cloud                           │
│  Dashboard    │  Streamlit App                             │
└────────────────────────────────────────────────────────────┘
```

---

## Hands-On Exercise

### Task: Deploy Your Own App

1. Fork the repository: https://github.com/VenkateswarluPudur/mlops
2. Make a small change to `app.py`
3. Push to your fork
4. Deploy to Streamlit Cloud
5. Share your URL with the class

### Bonus Tasks

- [ ] Add a new feature to the Streamlit app
- [ ] Create a GitHub Issue for a feature request
- [ ] Make a Pull Request with improvements

---

## Resources

| Resource | Link |
|----------|------|
| GitHub Repo | https://github.com/VenkateswarluPudur/mlops |
| Live App | https://d8vz9zebcwgfornj2e2ud3.streamlit.app |
| Streamlit Docs | https://docs.streamlit.io |
| GitHub Actions Docs | https://docs.github.com/en/actions |

---

## Q&A

Common questions:

**Q: Is Streamlit Cloud really free?**
A: Yes! Free tier includes 1GB memory, unlimited public apps.

**Q: What if my app needs more resources?**
A: Upgrade to paid tier or use Hugging Face Spaces (free, more resources).

**Q: Can I use a custom domain?**
A: Yes, on paid Streamlit plans.

**Q: How do I update my deployed app?**
A: Just push to GitHub - it auto-deploys!

---

*Day 5 Complete! 🎉*
