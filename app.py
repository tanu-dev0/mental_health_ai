from transformers import pipeline

# 1. Load the 'Brain' (A model trained on 6 emotions)
print("Loading the AI model... please wait.")
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# 2. Get user input
user_text = input("\nDescribe how you are feeling today: ")

# 3. Analyze
result = classifier(user_text)

# 4. Show the result
label = result[0]['label']
score = result[0]['score']

print(f"\n--- Analysis ---")
print(f"Detected Emotion: {label.upper()}")
print(f"Confidence: {score:.2%}")