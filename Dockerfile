# =============================================================================
# Woolworths NZ Demand Forecasting — Dockerfile
# =============================================================================


FROM python:3.10-slim

# Working directory inside the container
WORKDIR /workspace

# Install system dependencies that some Python packages need to compile
# pmdarima needs gfortran + BLAS/LAPACK on ARM
# matplotlib needs build tools
# curl is useful for debugging network issues
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first (avoids many install issues)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements (Docker caches this layer — only reinstalls if file changes)
COPY requirements.txt .

# Install all Python libraries
# --no-cache-dir keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports for Jupyter (8888) and Streamlit (8501)
EXPOSE 8888 8501

# Default command: start Jupyter notebook
# Token + password disabled for local dev only — never do this in production
CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=", \
     "--NotebookApp.password=", \
     "--notebook-dir=/workspace"]
