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
import re # Added for domain cleaning

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

class BusinessDescriptionRequest(BaseModel):
    business_description: str

class DomainSuggestion(BaseModel):
    domain: str
    confidence: float

class DomainSuggestionsResponse(BaseModel):
    suggestions: List[DomainSuggestion]
    status: str
    message: Optional[str] = None

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

@app.post("/suggest-domains", response_model=DomainSuggestionsResponse)
async def suggest_domains(request: BusinessDescriptionRequest):
    """
    Generate domain name suggestions based on business description.
    
    This endpoint provides a simplified interface for domain name generation
    with built-in safety guardrails.
    """
    try:
        # Safety check using the business description
        is_safe, safety_message = safety_guardrails.filter_unsafe_content(
            request.business_description
        )
        
        if not is_safe:
            return DomainSuggestionsResponse(
                suggestions=[],
                status="blocked",
                message="Request contains inappropriate content"
            )
        
        # Generate domain names using the model
        generated_domains = generate_domain_names_with_model(
            request.business_description,
            num_sequences=3,  # Generate 3 suggestions as per example
            temperature=0.8
        )
        
        # Process and validate generated domains
        suggestions = []
        for domain in generated_domains:
            # Clean up the domain (remove any extra text, ensure .com format)
            clean_domain = clean_domain_name(domain)
            
            # Validate the generated domain
            is_valid, validation_message = safety_guardrails.validate_generated_domain(
                clean_domain, request.business_description
            )
            
            if is_valid and clean_domain:
                # Calculate confidence score based on domain quality
                confidence = calculate_domain_confidence(clean_domain, request.business_description)
                
                suggestions.append(DomainSuggestion(
                    domain=clean_domain,
                    confidence=confidence
                ))
        
        # Sort by confidence score (highest first)
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        return DomainSuggestionsResponse(
            suggestions=suggestions,
            status="success"
        )
        
    except Exception as e:
        return DomainSuggestionsResponse(
            suggestions=[],
            status="error",
            message=f"Generation failed: {str(e)}"
        )

def clean_domain_name(domain: str) -> str:
    """Clean and format domain name."""
    # Remove any extra text and keep only the domain part
    domain = domain.strip()
    
    # Remove common prefixes/suffixes that might be added by the model
    domain = re.sub(r'^domain:\s*', '', domain, flags=re.IGNORECASE)
    domain = re.sub(r'^suggestion:\s*', '', domain, flags=re.IGNORECASE)
    domain = re.sub(r'^name:\s*', '', domain, flags=re.IGNORECASE)
    
    # Ensure it has a .com extension if no TLD is present
    if not re.search(r'\.(com|org|net|edu|gov|mil|int)$', domain, re.IGNORECASE):
        domain = domain + '.com'
    
    # Remove any spaces and special characters that aren't valid in domains
    domain = re.sub(r'[^\w\-\.]', '', domain)
    
    # Ensure it starts with a letter or number
    if not re.match(r'^[a-zA-Z0-9]', domain):
        domain = 'domain' + domain
    
    return domain.lower()

def calculate_domain_confidence(domain: str, business_description: str) -> float:
    """Calculate confidence score for a domain name."""
    confidence = 0.5  # Base confidence
    
    # Check domain length (prefer shorter domains)
    if len(domain) <= 15:
        confidence += 0.1
    elif len(domain) <= 20:
        confidence += 0.05
    else:
        confidence -= 0.1
    
    # Check if domain contains relevant keywords from business description
    business_words = set(re.findall(r'\b\w+\b', business_description.lower()))
    domain_words = set(re.findall(r'\b\w+\b', domain.lower()))
    
    # Calculate keyword relevance
    relevant_words = business_words.intersection(domain_words)
    if relevant_words:
        confidence += min(0.2, len(relevant_words) * 0.05)
    
    # Check for common domain patterns
    if re.match(r'^[a-z0-9]+\.com$', domain):
        confidence += 0.1
    
    # Penalize domains that are too generic
    generic_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    if domain_words.intersection(generic_words):
        confidence -= 0.05
    
    # Ensure confidence is between 0.0 and 1.0
    return max(0.0, min(1.0, confidence))

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
