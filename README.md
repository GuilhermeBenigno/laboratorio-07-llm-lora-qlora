# LLM Fine-Tuning com LoRA e QLoRA

Projeto do Laboratório 07 com objetivo de realizar fine-tuning de um modelo de linguagem usando técnicas eficientes.

---

##  Objetivo

* Gerar dataset sintético
* Aplicar QLoRA (quantização 4-bit)
* Realizar fine-tuning com LoRA

---

##  Modelo

* Modelo base: Phi-2
* Dataset: gerado com API da OpenAI

---

##  Tecnologias

* Transformers
* PEFT (LoRA)
* BitsAndBytes (QLoRA)
* TRL (SFTTrainer)

---

##  Como executar

### 1. Instalar dependências

```
pip install -r requirements.txt
```

### 2. Configurar API

```
export OPENAI_API_KEY="sua_chave"
```

### 3. Gerar dataset

```
python data/generate_dataset.py
```

### 4. Treinar modelo

```
python src/train.py
```

---

##  Estrutura

```
data/
src/
```

---

##  Observação

O modelo Phi-2 foi utilizado devido às restrições de acesso do Llama 2.

---

##  Uso de IA

Partes geradas/complementadas com IA, revisadas por Guilherme Gomes.
