#!/usr/bin/env python3
"""
Test the Coconut inference script with examples
"""

from coconut_inference import CoconutInference

def test_coconut_inference():
    """Test both reasoning and standard modes"""
    
    # Initialize with your model
    model_path = "coconut_model_hf"  # or "agurung/coconut-qwen2.5-7b" when uploaded
    coconut = CoconutInference(model_path)
    
    print("="*60)
    print("COCONUT INFERENCE TEST")
    print("="*60)
    
    # Test 1: Standard generation (no latent tokens)
    print("\n1. STANDARD MODE TEST:")
    standard_prompt = "What is the capital of France?"
    result1 = coconut.generate(standard_prompt, max_new_tokens=50)
    print(f"Prompt: {standard_prompt}")
    print(f"Result: {result1}")
    
    # Test 2: Reasoning mode (with latent tokens)
    print("\n2. REASONING MODE TEST:")
    reasoning_prompt = """Question: If a train travels 120 miles in 2 hours, what is its average speed?

Let me think step by step:
<|latent|> <|latent|> <|latent|>

In summary:"""
    
    result2 = coconut.generate(reasoning_prompt, max_new_tokens=100)
    print(f"Prompt: {reasoning_prompt}")
    print(f"Result: {result2}")
    
    # Test 3: Complex reasoning
    print("\n3. COMPLEX REASONING TEST:")
    complex_prompt = """Solve this problem step by step:

A store has 100 apples. They sell 30% on Monday, 25% of the remaining on Tuesday. How many are left?

Step 1: <|latent|>
Step 2: <|latent|>  
Step 3: <|latent|>

Final answer:"""
    
    result3 = coconut.generate(complex_prompt, max_new_tokens=150)
    print(f"Prompt: {complex_prompt}")
    print(f"Result: {result3}")

if __name__ == "__main__":
    test_coconut_inference()