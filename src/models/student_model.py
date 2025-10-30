import torch
import json
import re
import google.generativeai as genai


class Student:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("models/gemma-3-4b-it")
    

    def build_prompt(self, record):
        query = f"""
            You are an experienced **agricultural advisor** helping Indian farmers.
            The entire conversation, reasoning, and output must be **in Hindi language** using **Devanagari script**. And nothing should be generated in English at all.
            Each key's value must be a single, complete paragraph written entirely in Hindi.
            Do not use any English words, English numerals, or brackets “()” anywhere in your answer. category should be just in hindi without translation meaning.

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

    def build_prompt2(self, record):
        query = f"""
            You are an experienced agricultural advisor who provides guidance to Indian farmers **only in the Hindi language**, using the **Devanagari script**.

            Important Instructions (Must Follow Strictly):
            1. Respond only in Hindi language using Devanagari script.
            2. Do not use any English words, English numerals, or brackets “()” anywhere in your answer.
            3. Output must strictly follow the JSON structure shown below — no extra text outside the JSON.
            4. Each key's value must be a single, complete paragraph written entirely in Hindi.
            5. Do not include any English translations, terms, or explanations.
            6. Pesticide or fertilizer names must also be written in **Hindi transliteration**  
            (for example: write “प्रोपिकोनाजोल” instead of “Propiconazole”).

            **Example JSON format:**
            {{
            "category": "रोग",
            "rationale": "यह समस्या अत्यधिक नमी और फफूंद के कारण उत्पन्न हुई है।",
            "advisor_response": "रोग नियंत्रण के लिए खेत में जल निकासी सुनिश्चित करें और उचित फफूंदनाशी का छिड़काव करें।"
            }}

            Now, based on the following context and the farmer's question, generate a JSON response **exactly in the same format** as above.  
            Do not include any extra commentary or markdown formatting.

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
        response = self.model.generate_content(prompt)
        text = response.text.strip()
        print(text)

        # Try to extract the JSON part only
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        json_str = json_match.group(0) if json_match else text

        failed = False
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            failed = True
            parsed = {
                "category": "[FAILED]",
                "advisor_response": "[NO_OUTPUT]",
                "rationale": "[NO_REASONING]"
            }

        return {
            "pred_category": parsed.get("category", "").strip(),
            "pred_advisor_response": parsed.get("advisor_response", "").strip(),
            "pred_rationale": parsed.get("rationale", "").strip(),
            "raw_response": text,
            "failed": failed, 
        }