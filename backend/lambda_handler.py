"""
Lambda handler for RF Classifier API

This module wraps the FastAPI application for AWS Lambda using Mangum.
Mangum translates between Lambda's event format and ASGI (FastAPI).
"""

from mangum import Mangum
from main import app

# Create Lambda handler by wrapping FastAPI app
handler = Mangum(app, lifespan="off")
