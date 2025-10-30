import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv

# load_dotenv()

# Resolve absolute path to .env file
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
print("Looking for .env at:", env_path)

# Load .env file
if not load_dotenv(env_path):
    raise FileNotFoundError(f".env file not found at {env_path}")
# Debug print
print("Loaded GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set your GEMINI_API_KEY environment variable.")
    

'''
•	C1 – Classification: Identify the category (e.g., pest, irrigation, weather).
•	G1 – Generation: Generate advisor_response from farmer_query.
•	G3 – Rationale: Generate or evaluate the rationale explaining why a response is suitable.
'''


def devanagari_to_ascii(text: str) -> str:
    """Convert Devanagari digits to ASCII digits for parsing."""
    return text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))

def ascii_to_devanagari(data):
    """
    Recursively convert ASCII digits (0–9) to Devanagari digits (०–९)
    in all string or numeric values within a Python dictionary or list.
    """
    trans_map = str.maketrans("0123456789", "०१२३४५६७८९")
    # Case 1️⃣ — if the data is a dictionary
    if isinstance(data, dict):
        return {k: ascii_to_devanagari(v) for k, v in data.items()}

    # Case 2️⃣ — if it's a list
    elif isinstance(data, list):
        return [ascii_to_devanagari(item) for item in data]

    # Case 3️⃣ — if it's an integer or float
    elif isinstance(data, (int, float)):
        # Convert numeric types to string before translation, then back
        # e.g. 25 → "२५" (string) to preserve script rendering
        return str(data).translate(trans_map)

    # Case 4️⃣ — if it's a string containing ASCII digits
    elif isinstance(data, str):
        return data.translate(trans_map)

    # Case 5️⃣ — anything else (e.g., None, bool)
    else:
        return data



def generate_synthetic_data(api_key: str, num_batches: int = 3, samples_per_batch: int = 5, output_path: str = "data/synthetic_dataset.jsonl"):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-pro")

    prompt_template = """
            You are an agricultural expert **simulating your reasoning process** to assist Indian farmers.

            Generate {samples_per_batch} realistic farmer-advisor examples **with explicit model thinking**.

            Each entry must follow this exact JSON schema:
            {{
            "id": int,
            "crop": str,
            "region": str,
            "season": str,
            "rainfall_mm": float,                 // Rainfall during the last week (in millimeters)
            "avg_temp_c": float,                  // Average air temperature during the last week (°C)
            "humidity_percent": float,            // Average relative humidity (%)
            "soil_type": str,                     // Soil type (e.g., Loamy, Clayey, Sandy)
            "soil_temp_c": float,                 // Soil temperature (°C)
            "soil_pH": float,                     // Soil pH value
            "fertilizer_used": str,               // Recently applied fertilizer, if any
            "pesticide_used": str,                // Recently applied pesticide, if any
            "plant_height_cm": float,             // Current average plant height in cm
            "farmer_query": str,                  // Farmer's natural query or concern
            "model_thinking": str,                // Model's internal reasoning tokens (step-by-step explanation)
            "advisor_response": str,              // Final actionable advisory given to the farmer
            "category": str                       // Classification of the problem (e.g., Pest, Disease, Irrigation, Nutrient Deficiency, Weather, etc.)
            }}

            Guidelines:
            - Entire generated data should be in hindi strictly. All the numbers and words should be in hindi. Strictly no use of english words or numbers not even in brackets to tel the word meaning in engligh. 
            - Keep farmer queries natural, realistic, and regionally appropriate.
            - Invent reasonable environmental conditions (rainfall, temperature, humidity, soil, etc.) that make sense for the crop and season.
            - The **model_thinking** field must describe the reasoning steps in a highly elaborated way that is used to arrive at the final advice, referencing the environmental parameters where relevant.
            - The **advisor_response** should be factual, and actionable, mentioning clear interventions (dosages, sprays, timing, or irrigation steps).
            - Use realistic Indian regions (Punjab, Bihar, Tamil Nadu, Maharashtra, etc.) and valid crop seasons (Rabi, Kharif).
            - Cover diverse categories: Pest, Disease, Irrigation, Nutrient Deficiency, Weather, Weed Management, Soil Health, Harvest & Storage.

            Output Rules:
            - Output only **newline-separated JSON dictionaries**, one per record.
            - Do NOT include arrays, markdown code blocks, or any text outside of JSON.
            - Each dictionary must be fully self-contained and valid JSON.
    """

        ## for checking the performance later
        # "task_labels": {{
        #     "classification": str,      // same as category
        #     "generation_target": str,   // same as advisor_response
        #     "rationale_target": str     // same as model_thinking
        # }}

    global_id = 1 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as outfile:
        for batch in range(num_batches):
            print(f"\nGenerating batch {batch + 1}/{num_batches}...")
            prompt = prompt_template.format(samples_per_batch=samples_per_batch)
            try:
                response = model.generate_content(prompt)
                text = response.text.strip()
                
                json_blocks = re.findall(r'\{.*?\}', text, re.DOTALL)
                print(json_blocks)
                for block in json_blocks:
                    ascii_block = devanagari_to_ascii(block)
                    print("blockwise: ")
                    print(block)
                    print(type(block))
                    dict_block = json.loads(ascii_block)
                    dict_block = ascii_to_devanagari(dict_block)
                    dict_block["id"] = global_id
                    global_id += 1

                    json.dump(dict_block, outfile, ensure_ascii=False)
                    outfile.write("\n")
                    
                    # print("\njson block:\n ")
                    # print(type(dict_block))
                    # print(dict_block)
                    # print("\nnew json block:\n ")
                    # final_str = devanagari_json[:-1] + f', "id": {global_id}' + devanagari_json[-1:]
                    # print(final_str)
                    # outfile.write(final_str) #.strip())


                # outfile.write(text.strip())


                # if "```" in text:
                #     text = text.replace("```json", "").replace("```", "").strip()

                # #Split by newlines, parse each line individually
                # json_lines = [line.strip() for line in text.splitlines() if line.strip().startswith("{") and line.strip().endswith("}")]

                # parsed_records = []
                # for line in json_lines:
                #     try:
                #         rec = json.loads(line)
                #         parsed_records.append(rec)
                #     except json.JSONDecodeError:
                #         print(f"Skipping malformed line: {line[:100]}...")
                #         continue

                # for record in parsed_records:
                #     json.dump(record, outfile, ensure_ascii=False)
                #     outfile.write("\n")

                

                print(f"Saved  records from batch {batch + 1}") #{len(text)}

            except Exception as e:
                print(f"Error in batch {batch + 1}: {e}")

    print(f"\n Synthetic data saved to: {output_path}")



if __name__ == "__main__":
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found")
    
    generate_synthetic_data(GEMINI_API_KEY)