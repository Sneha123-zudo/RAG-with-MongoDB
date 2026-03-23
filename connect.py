from pymongo import MongoClient

uri = "mongodb+srv://sneha272:Sneha%40272@cluster0.xarp0ac.mongodb.net/?appName=Cluster0"

client = MongoClient(uri)

db = client["myDB"]
print("Connected successfully!")