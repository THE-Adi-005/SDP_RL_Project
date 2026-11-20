# Datasets and Exploratory Data Analysis (EDA)

For the empirical evaluation of the CascadeRoute framework, we selected three distinct reasoning datasets. These datasets were chosen to cover a diverse spectrum of cognitive tasks—ranging from arithmetic logic to commonsense deduction—ensuring that the joint routing policy is robust across varying degrees of complexity.

## 1. GSM8K (Grade School Math 8K)
*   **Domain:** Mathematical Reasoning
*   **Description:** A dataset of 8.5K high-quality linguistically diverse grade school math word problems. Solving these problems typically requires 2 to 8 steps of elementary arithmetic operations.
*   **Role in CascadeRoute:** Serves as the primary benchmark for multi-step algorithmic reasoning where strategies like Chain-of-Thought (CoT) traditionally excel.
*   **Source / Link:** [OpenAI GSM8K (HuggingFace)](https://huggingface.co/datasets/openai/gsm8k)

## 2. LogiQA (Logical Question Answering)
*   **Domain:** Formal Logical Deduction
*   **Description:** Constructed from the National Civil Servants Examination of China, LogiQA requires complex logical reasoning, reading comprehension, and deductive inference.
*   **Role in CascadeRoute:** Represents the upper bound of problem complexity in our evaluation. It rigorously tests the framework's ability to escalate to high-tier strategies (e.g., Tree-of-Thoughts) and fallback models when facing out-of-distribution logical structures.
*   **Source / Link:** [LogiQA Dataset (HuggingFace)](https://huggingface.co/datasets/lucasmccabe-lmi/logiqa) 
*(Note: Due to original repository deprecation, we utilized the `ARC-Challenge` as a logic-heavy fallback in standard runs).*

## 3. CommonsenseQA
*   **Domain:** Commonsense Reasoning
*   **Description:** A multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers. The questions are conceptual and often rely on implicit human knowledge.
*   **Role in CascadeRoute:** Acts as a counterbalance to GSM8K and LogiQA. Commonsense questions often do not benefit heavily from deep reasoning trees (ToT/GoT), allowing the LinUCB bandit to demonstrate frugality by routing these to `3B_Direct` or `7B_Direct`.
*   **Source / Link:** [CommonsenseQA (HuggingFace)](https://huggingface.co/datasets/tau/commonsense_qa)

## Dataset Distribution in Experiments
In our primary experiments (e.g., the 3,700 problem run), the data is distributed roughly equally across the three domains. The contextual bandit's state vector embeds the semantic properties of each domain (via SentenceTransformers), allowing it to implicitly learn that, for example, math queries require CoT while commonsense queries require Direct prompting, all without explicit domain labels.
