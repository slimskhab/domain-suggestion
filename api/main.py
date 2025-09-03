"""
API Deployment for Domain Name Generation

Optional API endpoint deployment for the domain name suggestion model.
"""

import os
import json
import torch
import pandas as pd
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import uvicorn
from datetime import datetime

# Import our modules
import sys
sys.path.append('src')
from safety.safety_guardrails import SafetyGuardrails
from evaluation.run_evaluation import LLMJudgeEvaluator

# API Models
class DomainNameRequest(BaseModel):
    prompt: str
    business_type: Optional[str] = None
    num_suggestions: Optional[int] = 5
    temperature: Optional[float] = 0.8

class DomainNameResponse(BaseModel):
    suggestions: List[str]
    safety_check: Dict
    evaluation_scores: Optional[Dict] = None
    timestamp: str

class SafetyCheckRequest(BaseModel):
    prompt: str
    business_type: Optional[str] = None

class SafetyCheckResponse(BaseModel):
    is_safe: bool
    risk_score: float
    flagged_keywords: List[str]
    reason: str

# Initialize FastAPI app
app = FastAPI(
    title="Domain Name Suggestion API",
    description="AI-powered domain name generation with safety guardrails",
    version="1.0.0"
)

# Global variables for model and components
model = None
tokenizer = None
safety_guardrails = None
evaluator = None

def load_model():
    """Load the trained model and components."""
    global model, tokenizer, safety_guardrails, evaluator
    
    print("Loading model and components...")
    
    # Load model and tokenizer
    model_path = "models/baseline/final"  # Update with your model path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Initialize safety guardrails
    safety_guardrails = SafetyGuardrails()
    
    # Initialize evaluator (optional, for evaluation endpoints)
    try:
        evaluator = LLMJudgeEvaluator()
    except Exception as e:
        print(f"Warning: Could not initialize evaluator: {e}")
        evaluator = None
    
    print("✅ Model and components loaded successfully")

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    load_model()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Domain Name Suggestion API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "safety_guardrails": safety_guardrails is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate", response_model=DomainNameResponse)
async def generate_domain_names(request: DomainNameRequest):
    """Generate domain name suggestions."""
    try:
        # Safety check
        is_safe, safety_message = safety_guardrails.filter_unsafe_content(
            request.prompt, request.business_type
        )
        
        if not is_safe:
            raise HTTPException(
                status_code=400,
                detail=f"Request blocked by safety guardrails: {safety_message}"
            )
        
        # Generate domain names
        suggestions = generate_domain_names_with_model(
            request.prompt,
            num_sequences=request.num_suggestions,
            temperature=request.temperature
        )
        
        # Validate generated domains
        validated_suggestions = []
        for suggestion in suggestions:
            is_valid, validation_message = safety_guardrails.validate_generated_domain(
                suggestion, request.prompt
            )
            if is_valid:
                validated_suggestions.append(suggestion)
        
        # Create safety check summary
        safety_check = {
            "input_safe": is_safe,
            "safety_message": safety_message,
            "validated_suggestions": len(validated_suggestions),
            "total_suggestions": len(suggestions)
        }
        
        # Optional: Evaluate suggestions if evaluator is available
        evaluation_scores = None
        if evaluator and validated_suggestions:
            try:
                # Evaluate first suggestion as example
                eval_result = evaluator.evaluate_domain_name(
                    request.prompt,
                    validated_suggestions[0],
                    request.business_type or "general"
                )
                evaluation_scores = {
                    "relevance": eval_result.relevance_score,
                    "creativity": eval_result.creativity_score,
                    "uniqueness": eval_result.uniqueness_score,
                    "safety": eval_result.safety_score,
                    "overall": eval_result.overall_score
                }
            except Exception as e:
                print(f"Evaluation failed: {e}")
        
        return DomainNameResponse(
            suggestions=validated_suggestions,
            safety_check=safety_check,
            evaluation_scores=evaluation_scores,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/safety-check", response_model=SafetyCheckResponse)
async def check_safety(request: SafetyCheckRequest):
    """Check if a prompt is safe for domain name generation."""
    try:
        safety_check = safety_guardrails.check_input_safety(
            request.prompt, request.business_type
        )
        
        return SafetyCheckResponse(
            is_safe=safety_check.is_safe,
            risk_score=safety_check.risk_score,
            flagged_keywords=safety_check.flagged_keywords,
            reason=safety_check.reason
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Safety check failed: {str(e)}")

@app.get("/business-types")
async def get_business_types():
    """Get available business types."""
    business_types = [
        "tech_startup", "restaurant", "consulting", "ecommerce", 
        "healthcare", "education", "finance", "real_estate",
        "creative_agency", "manufacturing", "retail", "services"
    ]
    
    return {
        "business_types": business_types,
        "count": len(business_types)
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "device": str(next(model.parameters()).device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "safety_guardrails": safety_guardrails is not None,
        "evaluator": evaluator is not None
    }

def generate_domain_names_with_model(prompt: str, num_sequences: int = 5, temperature: float = 0.8) -> List[str]:
    """Generate domain names using the loaded model."""
    if model is None or tokenizer is None:
        raise ValueError("Model and tokenizer must be loaded")
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    generation_config = GenerationConfig(
        max_length=50,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=num_sequences,
        pad_token_id=tokenizer.eos_token_id
    )
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )
    
    # Decode outputs
    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        # Extract the generated part (after the prompt)
        if " -> " in text:
            generated_part = text.split(" -> ")[-1]
            generated_texts.append(generated_part.strip())
        else:
            generated_texts.append(text.strip())
    
    return generated_texts

# Example usage and testing
if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
