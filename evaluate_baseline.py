import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key="---")

def get_openai_response(gt, pred):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Your job is to evaluate if the ground-truth and prediction are same/similar. Provide only Yes or No answer as JSON of the following structure {'answer': ''} without any explanation."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Ground-truth:{gt}\nPrediction:{pred}",
                    }
                ]
            }
        ],
        temperature=1,
        max_tokens=10,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "json_object"
        }
    )
    return response.choices[0].message.content

# Read the input CSV file
input_file = './data.csv'
df = pd.read_csv(input_file, sep=",")
model = "qwen"

# Create a list to store all QA pairs
eval_data = []

# Process each row and call the OpenAI API
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    data_id = row['ID']
    gt = row['Answer']
    pred = row[f'Generated Answer({model})']
    try:
      api_response = get_openai_response(gt, pred)
      output = json.loads(api_response)
      output = output["answer"]
      eval_data.append({
          'id': data_id,
          'ground_truth': gt,
          'prediction': pred,
          'result': output
      })
    except Exception as e:
      print(f"Error processing row {index}: {e}")
      eval_data.append({
          'id': data_id,
          'ground_truth': gt,
          'prediction': pred,
          'result': ''
      })

# Create a new DataFrame from the QA data
eval_df = pd.DataFrame(eval_data)
# Save the results to a new CSV file
output_file = f'eval_df_{model}.csv'
eval_df.to_csv(output_file, index=False)
print(f"Processing complete. Results saved to {output_file}")
