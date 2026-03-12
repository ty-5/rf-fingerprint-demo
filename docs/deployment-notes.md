# AWS Deployment Notes - RF Fingerprint Demo

## Live URLs
- **Frontend**: AWS Amplify (auto-deploys from GitHub `main` branch)
- **Backend API**: `https://5mcnigjdil.execute-api.us-east-1.amazonaws.com`

## AWS Resources
- **ECR Repository**: `343541495829.dkr.ecr.us-east-1.amazonaws.com/rf-classifier-backend`
- **Lambda Function**: `rf-classifier-api` (us-east-1, 3008 MB, 300s timeout)
- **API Gateway**: HTTP API `5mcnigjdil` (us-east-1)

---

## Deploy Workflow

### Backend changes (rebuild + push):
```bash
cd C:/Users/Ty/rf-fingerprint-demo

# Build
docker build --platform linux/amd64 --provenance=false -t rf-classifier-backend -f backend/Dockerfile .

# Tag & push to ECR
docker tag rf-classifier-backend:latest 343541495829.dkr.ecr.us-east-1.amazonaws.com/rf-classifier-backend:latest
docker push 343541495829.dkr.ecr.us-east-1.amazonaws.com/rf-classifier-backend:latest

# Update Lambda
aws lambda update-function-code --function-name rf-classifier-api --image-uri 343541495829.dkr.ecr.us-east-1.amazonaws.com/rf-classifier-backend:latest --region us-east-1
```

### Frontend changes:
Just `git push` — Amplify auto-deploys from the `main` branch.

---

## Lessons Learned

### 1. `.pth` file extension conflict
Python's `site` module scans `/var/task/` for `.pth` files and reads them as text path-config files. The binary model weights file (`RF_Model_Weights_98%.pth`) caused a `UnicodeDecodeError` crash at startup.

**Fix**: Copy the weights into the container with a `.pt` extension:
```dockerfile
COPY RF_Model_Weights_98%.pth ${LAMBDA_TASK_ROOT}/model_weights.pt
```

### 2. FastAPI startup events don't run in Lambda
Mangum (the Lambda/ASGI adapter) uses `lifespan="off"`, which skips `@app.on_event("startup")`. The model was never loading.

**Fix**: Load the model at module level so it runs during Lambda's cold start and is cached for warm invocations:
```python
def _init_model():
    global model, device
    if model is not None:
        return
    # ... load model ...

_init_model()  # Called at module level
```

### 3. Lambda Function URLs returned 403 Forbidden
Despite correct resource policies (`AuthType: NONE`, public `InvokeFunctionUrl` permission), the Function URL returned 403. This appears to be an account-level restriction.

**Fix**: Use **API Gateway HTTP API** instead — no public access restrictions, same free tier.

### 4. Docker build flags required for Lambda
Without the correct flags, the image uses OCI manifest format which Lambda doesn't support.

**Fix**: Always build with:
```bash
docker build --platform linux/amd64 --provenance=false ...
```

### 5. Dockerfile build context must be project root
The model weights and supporting Python files (`CNN_Extended.py`, `configs.py`) live in the project root, not in `backend/`. Docker can't access files outside the build context.

**Fix**: Build from the project root and reference files with their full relative paths:
```bash
docker build -f backend/Dockerfile .
```
