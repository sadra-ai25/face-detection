import pickle

with open("src/ai/face_db.pkl", "rb") as f:
    data = pickle.load(f)

print("Type:", type(data))

if isinstance(data, dict):
    print("Number of samples:", len(data))
    print("Keys example:", list(data.keys())) 
elif isinstance(data, list):
    print("Number of samples:", len(data))
    print("First element type:", type(data[0]))
elif hasattr(data, "shape"):  
    print("Shape:", data.shape)
else:
    try:
        print("Length:", len(data))
    except:
        print("Object has no length attribute.")
