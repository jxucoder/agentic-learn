# MLE-bench agent: agentic-learn
# Extends the mlebench-env base image

FROM mlebench-env

ARG SUBMISSION_DIR=/home/submission
ARG LOGS_DIR=/home/logs
ARG CODE_DIR=/home/code
ARG AGENT_DIR=/home/agent

# Create required directories
RUN mkdir -p ${SUBMISSION_DIR} ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

# Copy agent code
COPY . ${AGENT_DIR}/

# Install the agent dependencies
RUN /opt/conda/bin/conda run -n mleb pip install \
    anthropic>=0.40.0 \
    openai>=1.50.0 \
    httpx>=0.27.0 \
    pyyaml>=6.0

# Make start script executable
RUN chmod +x ${AGENT_DIR}/start.sh

# Set environment variables
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
ENV LOGS_DIR=${LOGS_DIR}
ENV CODE_DIR=${CODE_DIR}
ENV AGENT_DIR=${AGENT_DIR}
