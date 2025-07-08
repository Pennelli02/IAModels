# in questo file tenteremo di fare prompt engineering utilizzando un modello in particolare
# (Salesforce/blip-vqa-base) questo serve principalmenteper impratichirmi
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

question = "how many dogs are in the picture? And if you fond one describe it"
inputs = processor(raw_image, question, return_tensors="pt")

out = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(out[0], skip_special_tokens=True))
# risposta 1 è giusta però lui di sua spontanea volontà non descrive il cane dà solo risposte concise

question2 = "Describe in detail the animal"
inputs = processor(raw_image, question2, return_tensors="pt")
out2 = model.generate(**inputs)
print(processor.decode(out2[0], skip_special_tokens=True))
# risposta 2 non lo è per niente

question3 = "Describe in detail the person"
inputs = processor(raw_image, question3, return_tensors="pt")
out3 = model.generate(**inputs)
print(processor.decode(out3[0], skip_special_tokens=True))

question4 = "why the dog is tan?"
inputs = processor(raw_image, question4, return_tensors="pt")
out4 = model.generate(**inputs)
print(processor.decode(out4[0], skip_special_tokens=True))

question5 = "This image is fake or real?"
inputs = processor(raw_image, question5, return_tensors="pt")
out5 = model.generate(**inputs)
print(processor.decode(out5[0], skip_special_tokens=True))

question6 = "Which is the detail demonstrate the reality of this image?"
inputs = processor(raw_image, question6, return_tensors="pt")
out6 = model.generate(**inputs)
print(processor.decode(out6[0], skip_special_tokens=True))

raw_image2 = Image.open("test_images/testGenerated.png").convert('RGB')
inputs = processor(raw_image2, question5, return_tensors="pt")
out7 = model.generate(**inputs)
print(processor.decode(out7[0], skip_special_tokens=True))

question7 = "Describe this image"
inputs = processor(raw_image2, question7, return_tensors="pt")
out8 = model.generate(**inputs)
print(processor.decode(out8[0], skip_special_tokens=True))

question8 = "WHy this image is it real?"
inputs = processor(raw_image2, question8, return_tensors="pt")
out9 = model.generate(**inputs)
print(processor.decode(out9[0], skip_special_tokens=True))

question9 = "Why this image is it fake?"
inputs = processor(raw_image2, question9, return_tensors="pt")
out10 = model.generate(**inputs)
print(processor.decode(out10[0], skip_special_tokens=True))
# non è molto efficiente
