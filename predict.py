from cog import BasePredictor, Input
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import replicate

class Predictor(BasePredictor):
    def setup(self):
        """Initialize the core conversational engine into GPU memory."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Base foundational intelligence model (No fine-tuning required)
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )

    def predict(
        self,
        task: str = Input(
            description="Choose the capability you want to run",
            choices=["chat", "generate_image"],
            default="chat"
        ),
        prompt: str = Input(description="Input text prompt or command for the model"),
        system_prompt: str = Input(
            description="System instructions (only used for chat task)",
            default="You are Aeges Core, a helpful and advanced AI orchestrator assistant."
        ),
        max_tokens: int = Input(description="Maximum tokens to generate for chat", default=512, ge=1, le=2048),
        temperature: float = Input(description="Sampling temperature for chat", default=0.7, ge=0.0, le=2.0)
    ) -> str:
        """Route user requests to the appropriate internal or external model capability."""
        
        if task == "chat":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False
                )
                
            response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response_text

        elif task == "generate_image":
            output = replicate.run(
                "black-forest-labs/flux-schnell",
                input={"prompt": prompt}
            )
            return str(output[0])
            
