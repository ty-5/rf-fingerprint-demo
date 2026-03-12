# AWS Deployment Guide - RF Fingerprint Demo

This guide will help you deploy your RF Fingerprint demo to AWS for **FREE** using:
- **AWS Lambda** (Container) for the backend
- **AWS Amplify** for the frontend

---

## Prerequisites

1. **AWS Account** (free tier eligible)
2. **Docker Desktop** installed on your computer
3. **AWS CLI** installed (for pushing Docker images)

---

## Part 1: Deploy Backend to AWS Lambda

### Step 1: Install Docker Desktop (if not already installed)

1. Download Docker Desktop: https://www.docker.com/products/docker-desktop/
2. Install and start Docker Desktop
3. Verify installation by running in terminal: `docker --version`

### Step 2: Install AWS CLI (if not already installed)

1. Download AWS CLI: https://aws.amazon.com/cli/
2. Install it
3. Verify: `aws --version`

### Step 3: Configure AWS CLI

1. Open terminal/command prompt
2. Run: `aws configure`
3. Enter:
   - **AWS Access Key ID**: (get from AWS Console → IAM → Users → Security Credentials)
   - **AWS Secret Access Key**: (from same place)
   - **Default region**: `us-east-1` (or your preferred region)
   - **Default output format**: `json`

### Step 4: Create ECR Repository (AWS Console)

1. Go to **AWS Console**: https://console.aws.amazon.com/
2. Search for **"ECR"** (Elastic Container Registry) in the top search bar
3. Click **"Create repository"**
4. Settings:
   - **Visibility**: Private
   - **Repository name**: `rf-classifier-backend`
   - **Tag immutability**: Disabled
   - **Scan on push**: Disabled (optional)
   - **Encryption**: AES-256 (default)
5. Click **"Create repository"**
6. **Keep this page open** - you'll need the repository URI

### Step 5: Build and Push Docker Image

1. Open terminal/command prompt
2. Navigate to your project backend folder:
   ```bash
   cd C:\Users\Ty\rf-fingerprint-demo\backend
   ```

3. **Get ECR login command** from AWS Console:
   - In your ECR repository page, click **"View push commands"**
   - Copy and run **Command 1** (login command)

   It will look like:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
   ```

4. **Build the Docker image** (Command 2):
   ```bash
   docker build -t rf-classifier-backend .
   ```
   ⏱️ This will take 5-10 minutes (downloading PyTorch, etc.)

5. **Tag the image** (Command 3 - copy from ECR push commands):
   ```bash
   docker tag rf-classifier-backend:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/rf-classifier-backend:latest
   ```

6. **Push to ECR** (Command 4):
   ```bash
   docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/rf-classifier-backend:latest
   ```
   ⏱️ This will take 5-10 minutes to upload

### Step 6: Create Lambda Function (AWS Console)

1. Go to **AWS Lambda**: https://console.aws.amazon.com/lambda/
2. Click **"Create function"**
3. Select **"Container image"**
4. Settings:
   - **Function name**: `rf-classifier-api`
   - **Container image URI**: Click **"Browse images"** → Select your ECR repository → Select `latest` tag
5. Click **"Create function"**
6. **Configure the function**:
   - Click **"Configuration"** tab → **"General configuration"** → **"Edit"**
   - **Memory**: 1024 MB (or higher for better performance)
   - **Timeout**: 30 seconds
   - **Ephemeral storage**: 512 MB (default is fine)
   - Click **"Save"**

### Step 7: Create Function URL (Simple API endpoint)

1. In your Lambda function page, go to **"Configuration"** → **"Function URL"**
2. Click **"Create function URL"**
3. Settings:
   - **Auth type**: NONE (public access)
   - **Configure cross-origin resource sharing (CORS)**: ✅ Check this box
   - **Allow origin**: `*`
   - **Allow methods**: `*`
   - **Allow headers**: `*`
4. Click **"Save"**
5. **Copy the Function URL** - you'll need this for the frontend!
   - It will look like: `https://abc123xyz.lambda-url.us-east-1.on.aws/`

### Step 8: Test the Backend

1. Open the Function URL in your browser (add `/` at the end)
2. You should see: `{"message": "RF Classifier API is running!", ...}`
3. Test the API docs: Add `/docs` to the URL
4. ✅ Backend is deployed!

---

## Part 2: Deploy Frontend to AWS Amplify

### Step 9: Update Frontend API Endpoint

**STOP!** Before deploying, we need to update the frontend to use your Lambda Function URL.

Tell me your **Lambda Function URL** from Step 7, and I'll update the frontend code for you.

### Step 10: Deploy to AWS Amplify (AWS Console)

1. Go to **AWS Amplify**: https://console.aws.amazon.com/amplify/
2. Click **"New app"** → **"Host web app"**
3. Select **"GitHub"** (or your git provider)
4. Click **"Authorize AWS Amplify"** to connect your GitHub account
5. Select:
   - **Repository**: `rf-fingerprint-demo`
   - **Branch**: `main`
6. Click **"Next"**
7. **Build settings**:
   - **App name**: `rf-fingerprint-demo`
   - Amplify will auto-detect it's a React app
   - Leave build settings as default
8. Click **"Next"** → **"Save and deploy"**
9. ⏱️ Wait 3-5 minutes for deployment
10. ✅ Your app is live! Click the URL to view it

---

## Estimated Costs

- **Lambda**: FREE (1M requests/month + 400,000 GB-seconds free tier)
- **ECR**: FREE (500 MB storage free tier - your image is ~1GB, so ~$0.10/month)
- **Amplify**: FREE (15 GB served/month free tier)

**Total**: Effectively FREE for demo usage! (~$0.10/month for ECR storage)

---

## Troubleshooting

### Lambda function times out
- Increase timeout in Configuration → General configuration (max 15 min)
- Increase memory (more memory = faster CPU)

### CORS errors in browser
- Make sure Function URL has CORS enabled
- Check Allow origin is `*` or your Amplify domain

### Model not loading
- Check Lambda logs in CloudWatch
- Verify model file is in the Docker image

---

## Next Steps

Once you give me your Lambda Function URL, I'll:
1. Update the frontend to call your API
2. Push the changes to GitHub
3. Amplify will auto-deploy the updated frontend
4. Your demo will be fully live! 🚀
