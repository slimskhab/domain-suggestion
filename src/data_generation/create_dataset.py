"""
Synthetic Dataset Creation for Domain Name Generation

This module creates synthetic datasets for training domain name suggestion models.
"""

import os
import json
import random
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
import yaml

# Business type templates for domain name generation
BUSINESS_TEMPLATES = {
    "tech_startup": {
        "description": "A technology startup company",
        "keywords": ["tech", "digital", "innovative", "smart", "future", "ai", "data", "cloud", "app", "mobile"],
        "examples": ["techflow.com", "innovatehub.com", "smartly.io", "futuretech.com", "datacloud.net"]
    },
    "restaurant": {
        "description": "A restaurant or food service business",
        "keywords": ["food", "kitchen", "dining", "cuisine", "bistro", "cafe", "grill", "pizza", "burger", "sushi"],
        "examples": ["tastebuds.com", "kitchencraft.com", "diningdelight.com", "cuisinecorner.com", "foodiehub.com"]
    },
    "consulting": {
        "description": "A consulting or advisory business",
        "keywords": ["consult", "advisory", "expert", "strategy", "solutions", "partners", "group", "firm", "associates"],
        "examples": ["expertconsult.com", "strategysolutions.com", "advisorypartners.com", "consultgroup.net"]
    },
    "ecommerce": {
        "description": "An e-commerce or online retail business",
        "keywords": ["shop", "store", "buy", "market", "retail", "commerce", "online", "mall", "boutique", "shopify"],
        "examples": ["shopwise.com", "buymart.com", "marketplace.com", "retailhub.com", "shoponline.net"]
    },
    "healthcare": {
        "description": "A healthcare or medical business",
        "keywords": ["health", "medical", "care", "wellness", "clinic", "therapy", "pharmacy", "dental", "vision"],
        "examples": ["healthcareplus.com", "medicalclinic.com", "wellnesscenter.com", "healthhub.net"]
    },
    "education": {
        "description": "An educational institution or training business",
        "keywords": ["learn", "education", "academy", "school", "training", "study", "campus", "university", "college"],
        "examples": ["learnacademy.com", "educationhub.com", "trainingcenter.com", "studyzone.net"]
    },
    "finance": {
        "description": "A financial services business",
        "keywords": ["finance", "bank", "credit", "loan", "invest", "wealth", "money", "capital", "funds"],
        "examples": ["financepro.com", "bankwise.com", "creditplus.com", "investhub.net"]
    },
    "real_estate": {
        "description": "A real estate business",
        "keywords": ["realty", "property", "home", "house", "estate", "real", "land", "build", "develop"],
        "examples": ["realtypro.com", "propertyhub.com", "homewise.com", "estateplus.net"]
    },
    "creative_agency": {
        "description": "A creative agency or design business",
        "keywords": ["creative", "design", "studio", "agency", "art", "brand", "marketing", "advertising"],
        "examples": ["creativestudio.com", "designhub.com", "brandagency.com", "artstudio.net"]
    },
    "manufacturing": {
        "description": "A manufacturing business",
        "keywords": ["manufacture", "factory", "industrial", "produce", "build", "make", "create", "industry"],
        "examples": ["manufacturepro.com", "factoryhub.com", "industrialplus.com", "buildwise.net"]
    },
    "retail": {
        "description": "A retail business",
        "keywords": ["retail", "shop", "store", "market", "mall", "boutique", "outlet", "shopify"],
        "examples": ["retailpro.com", "shopwise.com", "storehub.com", "marketplus.net"]
    },
    "services": {
        "description": "A general services business",
        "keywords": ["service", "help", "support", "assist", "care", "maintain", "repair", "clean"],
        "examples": ["servicepro.com", "helpwise.com", "supporthub.com", "assistplus.net"]
    }
}

# Complexity levels
COMPLEXITY_LEVELS = {
    "simple": {"max_length": 15, "word_count": (1, 2)},
    "medium": {"max_length": 25, "word_count": (2, 3)},
    "complex": {"max_length": 35, "word_count": (3, 4)}
}

# Domain suffixes
DOMAIN_SUFFIXES = [".com", ".net", ".io", ".co", ".org", ".biz", ".info"]

@dataclass
class DomainNameExample:
    """Represents a domain name example with metadata."""
    business_type: str
    complexity: str
    prompt: str
    domain_name: str
    input_text: str
    target_text: str
    quality_score: float = 0.0

class SyntheticDatasetGenerator:
    """Generates synthetic datasets for domain name generation."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize the dataset generator."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.business_templates = BUSINESS_TEMPLATES
        self.complexity_levels = COMPLEXITY_LEVELS
        self.domain_suffixes = DOMAIN_SUFFIXES
        
    def generate_domain_name(self, business_type: str, complexity: str) -> str:
        """Generate a synthetic domain name based on business type and complexity."""
        template = self.business_templates[business_type]
        keywords = template['keywords']
        
        if complexity == "simple":
            # Simple: keyword + suffix
            domain = random.choice(keywords) + random.choice(self.domain_suffixes)
        elif complexity == "medium":
            # Medium: keyword + modifier + suffix
            modifiers = ["hub", "pro", "plus", "max", "go", "now", "fast", "easy"]
            domain = random.choice(keywords) + random.choice(modifiers) + random.choice(self.domain_suffixes)
        else:  # complex
            # Complex: keyword + descriptive + suffix
            descriptives = ["solutions", "partners", "group", "systems", "services", "network", "platform"]
            domain = random.choice(keywords) + random.choice(descriptives) + random.choice(self.domain_suffixes)
        
        return domain
    
    def generate_prompt(self, business_type: str) -> str:
        """Generate a prompt for domain name generation."""
        template = self.business_templates[business_type]
        descriptions = [
            f"Generate a domain name for a {template['description']}.",
            f"Create a website domain for a {template['description']}.",
            f"Suggest a domain name for a {template['description']}.",
            f"Find a good domain name for a {template['description']}.",
            f"Come up with a domain name for a {template['description']}."
        ]
        return random.choice(descriptions)
    
    def create_dataset(self, num_samples: int = None) -> pd.DataFrame:
        """Create a synthetic dataset for domain name generation."""
  
        if num_samples <= 0:
            return pd.DataFrame()
        
        data = []
        
        for i in range(num_samples):
            # Randomly select business type and complexity
            business_type = random.choice(list(self.business_templates.keys()))
            complexity = random.choice(list(self.complexity_levels.keys()))
            
            # Generate prompt and domain name
            prompt = self.generate_prompt(business_type)
            domain_name = self.generate_domain_name(business_type, complexity)
            
            # Create example
            example = DomainNameExample(
                business_type=business_type,
                complexity=complexity,
                prompt=prompt,
                domain_name=domain_name,
                input_text=prompt,
                target_text=domain_name
            )
            
            data.append({
                'id': i,
                'business_type': example.business_type,
                'complexity': example.complexity,
                'prompt': example.prompt,
                'domain_name': example.domain_name,
                'input_text': example.input_text,
                'target_text': example.target_text
            })
        
        return pd.DataFrame(data)
    
    def create_balanced_dataset(self, samples_per_type: int = 100) -> pd.DataFrame:
        """Create a balanced dataset with equal samples per business type."""
        data = []
        id_counter = 0
        
        for business_type in self.business_templates.keys():
            for complexity in self.complexity_levels.keys():
                for _ in range(samples_per_type // len(self.complexity_levels)):
                    prompt = self.generate_prompt(business_type)
                    domain_name = self.generate_domain_name(business_type, complexity)
                    
                    data.append({
                        'id': id_counter,
                        'business_type': business_type,
                        'complexity': complexity,
                        'prompt': prompt,
                        'domain_name': domain_name,
                        'input_text': prompt,
                        'target_text': domain_name
                    })
                    id_counter += 1
        
        return pd.DataFrame(data)
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "domain_names_dataset.csv"):
        """Save the dataset to a file."""
        os.makedirs("data/synthetic_dataset", exist_ok=True)
        filepath = f"data/synthetic_dataset/{filename}"
        df.to_csv(filepath, index=False)
        print(f"✅ Dataset saved to {filepath}")
        return filepath
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict:
        """Analyze the generated dataset."""
        analysis = {
            "total_samples": len(df),
            "business_types": df['business_type'].value_counts().to_dict(),
            "complexity_levels": df['complexity'].value_counts().to_dict(),
            "avg_domain_length": df['domain_name'].str.len().mean(),
            "unique_domains": df['domain_name'].nunique(),
            "duplicate_rate": 1 - (df['domain_name'].nunique() / len(df))
        }
        
        return analysis

def main():
    """Main function to generate and save the dataset."""
    print("🚀 Generating synthetic dataset for domain name generation...")
    
    # Initialize generator
    generator = SyntheticDatasetGenerator()
    
    # Create dataset
    df = generator.create_dataset(1000)
    
    # Analyze dataset
    analysis = generator.analyze_dataset(df)
    print(f"📊 Dataset Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # Save dataset
    generator.save_dataset(df)
    
    print("✅ Dataset generation complete!")

if __name__ == "__main__":
    main()
