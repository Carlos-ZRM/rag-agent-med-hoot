"""
seed_pediatrics.py – Insert 20 pediatric true/false questions into MongoDB.

Usage:
    poetry run python seed_pediatrics.py

Env vars (optional):
    MONGO_URI  – default mongodb://localhost:27017
    MONGO_DB   – default health_kahoot
"""

import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "health_kahoot")
COLLECTION = "questions"

QUESTIONS: list[dict] = [
    {
        "text": "The anterior fontanelle in a newborn typically closes between 12 and 18 months of age.",
        "correct_answer": True,
        "category": "Anatomy",
        "difficulty": 1,
        "explanation": "The anterior fontanelle usually closes between 12–18 months, although some variation is normal.",
    },
    {
        "text": "Exclusive breastfeeding is recommended for the first 12 months of life.",
        "correct_answer": False,
        "category": "Nutrition",
        "difficulty": 1,
        "explanation": "WHO and AAP recommend exclusive breastfeeding for the first 6 months, with complementary foods introduced thereafter.",
    },
    {
        "text": "A normal respiratory rate for a newborn is between 30 and 60 breaths per minute.",
        "correct_answer": True,
        "category": "General",
        "difficulty": 1,
        "explanation": "Neonatal respiratory rates of 30–60 bpm are considered within the normal range.",
    },
    {
        "text": "The MMR vaccine is routinely administered at 2 months of age.",
        "correct_answer": False,
        "category": "Public Health",
        "difficulty": 2,
        "explanation": "The first dose of MMR is typically given at 12–15 months, with a second dose at 4–6 years.",
    },
    {
        "text": "Febrile seizures in children are most common between 6 months and 5 years of age.",
        "correct_answer": True,
        "category": "Pathology",
        "difficulty": 2,
        "explanation": "Febrile seizures peak in this age range and are the most common seizure type in childhood.",
    },
    {
        "text": "A heart rate of 180 bpm in a sleeping 3-year-old is within the normal range.",
        "correct_answer": False,
        "category": "General",
        "difficulty": 2,
        "explanation": "Normal resting heart rate for a 3-year-old is approximately 80–120 bpm; 180 bpm at rest suggests tachycardia.",
    },
    {
        "text": "Iron-deficiency anemia is the most common nutritional deficiency in children worldwide.",
        "correct_answer": True,
        "category": "Nutrition",
        "difficulty": 1,
        "explanation": "Iron deficiency remains the leading single-nutrient deficiency globally, especially among infants and toddlers.",
    },
    {
        "text": "Oral rehydration salts (ORS) are the first-line treatment for mild to moderate dehydration in pediatric gastroenteritis.",
        "correct_answer": True,
        "category": "Pharmacology",
        "difficulty": 1,
        "explanation": "ORS is the WHO-recommended first-line therapy, restoring fluids and electrolytes effectively.",
    },
    {
        "text": "Kawasaki disease primarily affects children under 5 years old and can lead to coronary artery aneurysms.",
        "correct_answer": True,
        "category": "Pathology",
        "difficulty": 3,
        "explanation": "About 80% of cases occur in children under 5. Without treatment, 15–25% may develop coronary artery abnormalities.",
    },
    {
        "text": "Aspirin is the preferred antipyretic for fever management in children of all ages.",
        "correct_answer": False,
        "category": "Pharmacology",
        "difficulty": 2,
        "explanation": "Aspirin is avoided in children due to the risk of Reye syndrome; acetaminophen or ibuprofen are preferred.",
    },
    {
        "text": "The Moro reflex is normally present at birth and disappears by 5–6 months of age.",
        "correct_answer": True,
        "category": "Anatomy",
        "difficulty": 2,
        "explanation": "The Moro (startle) reflex is a primitive reflex that typically integrates by 5–6 months.",
    },
    {
        "text": "A 2-year-old child should be able to speak in complete sentences of 5 or more words.",
        "correct_answer": False,
        "category": "General",
        "difficulty": 2,
        "explanation": "At 2 years, children typically use 2-word phrases. Sentences of 5+ words are expected closer to 4–5 years.",
    },
    {
        "text": "Vitamin K is administered to newborns to prevent hemorrhagic disease of the newborn.",
        "correct_answer": True,
        "category": "Pharmacology",
        "difficulty": 1,
        "explanation": "A single intramuscular dose of vitamin K at birth is standard practice to prevent vitamin K deficiency bleeding.",
    },
    {
        "text": "Pyloric stenosis typically presents with bilious vomiting in the first weeks of life.",
        "correct_answer": False,
        "category": "Pathology",
        "difficulty": 3,
        "explanation": "Pyloric stenosis causes non-bilious, projectile vomiting. Bilious vomiting suggests a more distal obstruction such as malrotation.",
    },
    {
        "text": "Children should receive their first dental visit by 1 year of age or within 6 months of the first tooth eruption.",
        "correct_answer": True,
        "category": "Public Health",
        "difficulty": 1,
        "explanation": "The American Academy of Pediatric Dentistry recommends the first dental visit by age 1 or within 6 months of the first tooth.",
    },
    {
        "text": "Croup is most commonly caused by respiratory syncytial virus (RSV).",
        "correct_answer": False,
        "category": "Pathology",
        "difficulty": 2,
        "explanation": "Croup is most commonly caused by parainfluenza viruses (types 1 and 3). RSV is the leading cause of bronchiolitis.",
    },
    {
        "text": "Neonatal jaundice appearing within the first 24 hours of life is always considered pathological.",
        "correct_answer": True,
        "category": "Pathology",
        "difficulty": 3,
        "explanation": "Jaundice in the first 24 hours suggests hemolysis or another pathological cause and requires urgent evaluation.",
    },
    {
        "text": "The recommended daily calcium intake for children aged 4–8 years is 1,000 mg.",
        "correct_answer": True,
        "category": "Nutrition",
        "difficulty": 2,
        "explanation": "The National Institutes of Health recommends 1,000 mg/day of calcium for children aged 4–8 years.",
    },
    {
        "text": "Intussusception is most common in adolescents between 12 and 16 years of age.",
        "correct_answer": False,
        "category": "Pathology",
        "difficulty": 3,
        "explanation": "Intussusception most commonly occurs in infants aged 6–36 months, not in adolescents.",
    },
    {
        "text": "Hand, foot, and mouth disease is caused by Coxsackievirus and is highly contagious among young children.",
        "correct_answer": True,
        "category": "Public Health",
        "difficulty": 1,
        "explanation": "HFMD is primarily caused by Coxsackievirus A16 and Enterovirus 71, spreading easily in daycare and preschool settings.",
    },
]


async def main():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[MONGO_DB]
    col = db[COLLECTION]

    existing = await col.count_documents({})
    print(f"Collection '{COLLECTION}' currently has {existing} documents.")

    result = await col.insert_many(QUESTIONS)
    print(f"Inserted {len(result.inserted_ids)} pediatric questions.")
    print("IDs:", [str(i) for i in result.inserted_ids])

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
