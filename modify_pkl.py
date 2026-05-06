import pickle

with open("src/ai/face_db.pkl", "rb") as f:
    data = pickle.load(f)

if isinstance(data, dict):
    new_data = {}
    for key, value in data.items():
        new_key = key.split("_")[0]
        new_data[new_key] = value

    with open("src/ai/face_db_new.pkl", "wb") as f:
        pickle.dump(new_data, f)

    print("✅ face_db_new.pkl created.")
    print("keys", list(new_data.keys())[:10])
else:
    print("❌ Error: the keys are out of the dictionary.")
