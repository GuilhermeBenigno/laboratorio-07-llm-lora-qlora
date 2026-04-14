import os
import json
from openai import OpenAI

client = OpenAI(api_key="SUA_API_KEY_AQUI")

os.makedirs("data", exist_ok=True)

def gerar_dataset(n=50):
    dataset = []

    for i in range(n):
        prompt = "Crie uma pergunta e resposta sobre programação para treinamento de IA. Retorne em JSON com 'prompt' e 'response'."

        resposta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        conteudo = resposta.choices[0].message.content

        try:
            item = json.loads(conteudo)
            dataset.append(item)
        except:
            print("Erro ao converter, pulando...")

    return dataset


if __name__ == "__main__":
    data = gerar_dataset()

    with open("data/dataset.jsonl", "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print("Dataset gerado com OpenAI!")
