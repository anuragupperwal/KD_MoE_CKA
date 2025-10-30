import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM


class Student:
    def __init__(self):
        """Initialize Gemma3n-2b student model."""
        self.model_name = "google/gemma-3n-e2b"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=None, torch_dtype=torch.bfloat16).to("cpu")


    def build_prompt(self, record):
        """
        Construct a reasoning-style bilingual prompt 
        that explicitly tells the model to respond in Hindi JSON.
        """
        query = f"""
        You are an experienced **agricultural advisor** helping Indian farmers.
        The entire conversation, reasoning, and output must be **in Hindi language** using **Devanagari script**.

        Given the following context and farmer's query, analyze the situation and provide:
        1. category – nature of the problem (कीट, रोग, पोषक तत्व की कमी, सिंचाई, मौसम, खरपतवार प्रबंधन, मृदा स्वास्थ्य, फसल कटाई)
        2. rationale – your step-by-step reasoning in Hindi
        3. advisor_response – factual, actionable advice in Hindi

        Return your response **strictly as a JSON object only** (no text outside JSON).

        Context:
        फसल (Crop): {record.get('crop', '')}
        क्षेत्र (Region): {record.get('region', '')}
        मौसम (Season): {record.get('season', '')}
        वर्षा (Rainfall, mm): {record.get('rainfall_mm', '')}
        औसत तापमान (°C): {record.get('avg_temp_c', '')}
        आर्द्रता (%): {record.get('humidity_percent', '')}
        मिट्टी का प्रकार: {record.get('soil_type', '')}
        मिट्टी का तापमान (°C): {record.get('soil_temp_c', '')}
        मिट्टी का pH: {record.get('soil_pH', '')}
        उपयोग की गई खाद: {record.get('fertilizer_used', '')}
        उपयोग किए गए कीटनाशक: {record.get('pesticide_used', '')}
        पौधे की ऊँचाई (सेमी): {record.get('plant_height_cm', '')}

        किसान का प्रश्न:
        {record.get('farmer_query', '')}
        """

        return query.strip()

    def predict(self, record):
        """Generate advisory JSON output for one record."""
        prompt = self.build_prompt(record)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        response = self.model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True,)

        text = self.tokenizer.decode(response[0], skip_special_tokens=True).strip()

        # Try to extract the JSON part only
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        json_str = json_match.group(0) if json_match else text

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            # fallback blank structure if still malformed
            parsed = {"category": "", "advisor_response": "", "rationale": ""}

        return {
            "pred_category": parsed.get("category", "").strip(),
            "pred_advisor_response": parsed.get("advisor_response", "").strip(),
            "pred_rationale": parsed.get("rationale", "").strip(),
            "raw_response": text,
        }