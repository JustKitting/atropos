FROM python:3.12-slim

# Install jj (Jujutsu VCS)
# Using the official cargo-binstall method for a static binary
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -L --proto '=https' --tlsv1.2 -sSf \
        https://github.com/jj-vcs/jj/releases/download/v0.35.0/jj-v0.35.0-x86_64-unknown-linux-musl.tar.gz \
        | tar xz -C /usr/local/bin && \
    chmod +x /usr/local/bin/jj && \
    apt-get purge -y curl && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Verify jj is installed
RUN jj --version

# Install Python dependencies
RUN pip install --no-cache-dir flask

WORKDIR /app

COPY server.py /app/server.py

EXPOSE 5003

CMD ["python", "server.py"]
