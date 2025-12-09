from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class LocalLLM:
    """
    Local Hugging Face LLM wrapper using Qwen2.5-0.5B-Instruct.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        print(f"Loading LLM model: {model_name} (this may take some time on first run)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",  # uses GPU if available, else CPU
        )

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Generate response text for a given prompt using Qwen.
        Returns only the newly generated text (prompt stripped).
        """
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        full_text = outputs[0]["generated_text"]
        # Remove prompt echo if present
        if full_text.startswith(prompt):
            full_text = full_text[len(prompt):]
        return full_text.strip()
