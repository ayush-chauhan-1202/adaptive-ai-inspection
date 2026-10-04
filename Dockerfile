# Multi-stage build: the "builder" stage has the full toolchain needed to
# install dependencies (including compiling any that need it); the final
# image only has the installed packages and the app code, which keeps the
# shipped image smaller and avoids shipping build tools into production.

FROM python:3.11-slim AS builder

WORKDIR /build

# System libs needed by opencv-python and Pillow at *install/import* time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

# Install into a venv under /opt/venv so the runtime stage can copy just
# that directory across, rather than re-resolving dependencies.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[serve]" && \
    pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu


FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# The venv copied from the builder stage already has the inspection package
# installed (non-editable), so nothing further needs to be pip-installed
# here - just the app's own source for ownership by appuser.
COPY src ./src

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080

# Cloud Run (and most container platforms) inject $PORT - respecting it
# rather than hardcoding 8080 is what makes this image portable across them.
CMD ["sh", "-c", "uvicorn inspection.api.main:app --host 0.0.0.0 --port ${PORT}"]
