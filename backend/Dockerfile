# Use AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.10

# Copy requirements first for better layer caching
COPY requirements_lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements_lambda.txt

# Copy application code
COPY main.py ${LAMBDA_TASK_ROOT}/
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/
COPY models/ ${LAMBDA_TASK_ROOT}/models/

# Copy model weights from parent directory
COPY ../RF_Model_Weights_98%.pth ${LAMBDA_TASK_ROOT}/
COPY ../CNN_Extended.py ${LAMBDA_TASK_ROOT}/
COPY ../configs.py ${LAMBDA_TASK_ROOT}/

# Set the Lambda handler
CMD ["lambda_handler.handler"]
