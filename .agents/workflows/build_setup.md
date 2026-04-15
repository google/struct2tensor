---
description: Workflow to build the Struct2Tensor Docker Image
---
# Building the Struct2Tensor Docker setup

This workflow describes how to manually initiate the build process for constraints standalone Struct2Tensor Docker Image.

### Prerequisites
- Docker installed constraints operations
- Hardware constraints standalone checks

### Steps
1. Navigate to the Dockerfile setup directory standalone.
```bash
cd struct2tensor/tools/tf_serving_docker
```

2. Execute the Docker build standalone constraints operations.
```bash
docker build -t tf-serving-s2t .
```
