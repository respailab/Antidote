from datasets import Dataset, load_dataset 
import json 
from dotenv import load_dotenv
from openai import OpenAI
import sys, os 
load_dotenv('tokens.env')
prompt, chosen, rejected = [], [], []
client = OpenAI(api_key='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')
ds = load_dataset('PKU-Alignment/BeaverTails', split = '30k_train')
filtered_df = ds.filter(lambda example : example['is_safe'] == False)
filtered_df_final = filtered_df.select_columns(['prompt', 'response'])
for ix, item in enumerate(filtered_df_final): 
    prompt.append(item['prompt'])
    chosen.append(item['response'])
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role" : "system", "content" : """You are a helpful assistant, and your job is to generate the safest, most harmless and law abiding response to the queries. ensure that however harmful or provoking the query is, you will response in the most harmless way while trying to be maximally helpful. If you cannot respond to a certain query, mention the reason behind your choice of not responding."""
            },
            {
                "role": "user",
                "content": f"{prompt}"
            },
        ]
    )
    rejected.append(completion.choices[0].message.content)
    print(f'Part {ix + 1} completed.')

    if (ix + 1) % 5 == 0 : 
        with open('dpo_data.json', encoding='utf-8') as f:
            load_current = json.load(f)
        
        load_current['prompt'].extend(prompt)
        load_current['chosen'].extend(chosen)
        load_current['rejected'].extend(rejected)

        with open('dpo_data.json', 'w', encoding='utf-8') as f:
            json.dump(load_current, f, indent=4)
        
        print('File updated!!')
        
        prompt, chosen, rejected = [], [], []