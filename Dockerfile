# Multi-stage build:
#   - builder stage installs the package (and its deps from PyPI) into /install.
#   - runtime stage gets only the installed Python packages — no apt cache,
#     no build tooling for anyone who lands a shell.
# Final image runs as uid 10001 (non-root). hostPath mounts elsewhere in the
# stack make root-in-container a real escalation primitive; this closes that.

# ---------------- Stage 1: builder ----------------
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md ./
COPY jeff/ ./jeff/

# All dependencies (including ensemble-client) come from PyPI.
RUN pip install --no-cache-dir --prefix=/install .

# ---------------- Stage 2: runtime ----------------
FROM python:3.12-slim AS runtime

# Non-root user. uid 10001 stays well clear of system uids; matching gid
# keeps `id` output clean. --no-create-home is fine because /home/jeff isn't
# used — we run from /app and write nothing else to the FS.
RUN groupadd --system --gid 10001 jeff \
    && useradd --system --uid 10001 --gid jeff --no-create-home --shell /usr/sbin/nologin jeff

# Copy installed site-packages + console scripts from the builder. Nothing
# else from stage 1 (no source tree under /build).
COPY --from=builder /install /usr/local

WORKDIR /app
COPY --chown=jeff:jeff jeff/ ./jeff/

ENV PYTHONUNBUFFERED=1
USER jeff

CMD ["python", "-m", "jeff"]
