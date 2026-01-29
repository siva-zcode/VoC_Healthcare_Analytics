# src/generate_dataset.py
import pandas as pd
import random
from faker import Faker
import numpy as np

fake = Faker()

# Parameters
num_patients = 1000
departments = ['Cardiology', 'Oncology', 'ER', 'Neurology', 'Orthopedics']
visit_types = ['In-person', 'Virtual']
ratings = [1, 2, 3, 4, 5]
feedback_texts = {
    1: ["Very poor service", "Long wait time", "Staff unhelpful", "Miscommunication", "Treatment delayed"],
    2: ["Poor experience", "Not satisfied", "Waiting too long", "Confusing instructions"],
    3: ["Average service", "Okay experience", "Nothing special"],
    4: ["Good experience", "Staff helpful", "Timely treatment"],
    5: ["Excellent service", "Very satisfied", "Friendly staff", "Quick response"]
}

# Generate data
data = []
for i in range(1, num_patients+1):
    rating = random.choice(ratings)
    data.append({
        'patient_id': i,
        'department': random.choice(departments),
        'visit_type': random.choice(visit_types),
        'rating': rating,
        'feedback_text': random.choice(feedback_texts[rating]),
        'date': fake.date_between(start_date='-1y', end_date='today')
    })

df = pd.DataFrame(data)

# Save CSV
df.to_csv('../data/synthetic_feedback.csv', index=False)
print("Synthetic dataset generated successfully!")
