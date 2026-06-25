# Official, lightweight Python runtime
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy ONLY the lockfile and pyproject.toml first
COPY pyproject.toml uv.lock ./

# Sync the dependencies
RUN uv sync --frozen --no-install-project

# Copy the rest of application code
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Command to run the Streamlit application
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]