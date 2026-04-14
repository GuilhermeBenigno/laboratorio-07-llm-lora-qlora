import json
import random

def gerar_dataset(n=50):
    data = []

    for i in range(n):
        prompt = f"Explique o conceito de programação número {i}"
        response = f"O conceito {i} envolve lógica, algoritmos e boas práticas."

        data.append({
            "prompt": prompt,
            "response": response
        })

    return data


if __name__ == "__main__":
    dataset = gerar_dataset()

    random.shuffle(dataset)

    with open("data/dataset.jsonl", "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print("Dataset gerado!")
