# Use Debian-based Python image for better TensorFlow support
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Remove build dependencies to reduce image size
RUN apt-get remove -y gcc g++ && apt-get autoremove -y && apt-get clean

# Create non-root user
RUN groupadd -g 1000 appuser && useradd -m -s /bin/bash -u 1000 -g appuser appuser

# Copy only essential application files
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser data/cleaned/ data/cleaned/
COPY --chown=appuser:appuser templates/ templates/
COPY --chown=appuser:appuser static/ static/

# Copy only the best model (final_model.h5) and model info
COPY --chown=appuser:appuser models/final_model.h5 models/
COPY --chown=appuser:appuser models/final_model_info.yaml models/

# Create logs directory
RUN mkdir -p /app/logs && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"] 