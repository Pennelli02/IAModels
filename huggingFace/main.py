from transformers import pipeline

# import torch

# test 1 e test 2
# classifier = pipeline('sentiment-analysis')
# result = classifier("I absolutely love using Hugging Face and PyTorch on GPU!")
# result = classifier(["I absolutely love using Hugging Face and PyTorch on GPU!",
#                   "I really hate it"])
# print(result)
# ----------------------------------------------------------
# test 3
# classifier = pipeline("zero-shot-classification")
# res = classifier("This is a course about Transformers of Hugging face",
#                  candidate_labels=["education", "politics", "health"])
# print(res)
# -----------------------------------------------------------
# test 4
# classifier = pipeline('text-generation')
# res = classifier("Write a message to my girlfriend to say hello and ask if she was ok")
# print(res)
# -------------------------------------------------------------
# test 5

# classifier = pipeline(
#     'text-generation',
#     model="distilgpt2",
#     pad_token_id=50256  # evita warning sul padding
# )
#
# results = classifier(
#     "In this course we will teach you",
#     max_new_tokens=100,
#     num_return_sequences=2
# )
#
# print(results)
# ----------------------------------------------------------------
# test 6

# unmasker = pipeline("fill-mask")
# result = unmasker("Hello <mask>, nice to meet you!", top_k=2)
# print(result)
# ------------------------------------------------------------------
# test 7

# ner = pipeline('ner', grouped_entities=True)
# results = ner(" My name is Pasquale and I work like broker in Via fratelli bandiera 1, Parma") # la mandiamo in vacca così
# print(results)
# cosa fa? Usa la pipeline di Named Entity Recognition (NER) di Hugging Face per estrarre entità dal testo, ad esempio:
#
# NOMI di persone
#
# LUOGHI
#
# ORGANIZZAZIONI
#
# DATE, ecc.

# ---------------------------------------------------------------------------------------------
# test 8

# question = pipeline("question-answering")
# answer = question(
#     question="what are you study?",
#     context="My name is Lorenzo and i study engineering with my friends"
# )
# print(answer)
# -------------------------------------------------------------------------------------------------
# test 9

# ci sono altre due funzioni
# # pipeline("summarization") che fa il riassunto
#
# # Pipeline di traduzione da inglese a italiano
# translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-it")
#
# # Frase da tradurre
# text = "I love using Hugging Face for natural language processing."
#
# # Traduzione
# translation = translator(text, max_length=100)
# print(translation)
# -------------------------------------------------------------------------------------------------
# tokenizer

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
inputs = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")
print(inputs)
