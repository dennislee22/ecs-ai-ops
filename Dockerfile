# Use the UBI Python 3.12 base image
FROM registry.redhat.io/ubi8/python-312

# Install git-lfs and enable it
RUN yum install -y git-lfs && \
    git lfs install && \
    yum clean all

# Set working directory to the S2I source directory
WORKDIR /opt/app-root/src

# Expose the port your app will use (8080 for default S2I Python)
EXPOSE 8080

# Default S2I scripts will handle copying source and running the app
# No CMD needed, S2I image already defines the run script
