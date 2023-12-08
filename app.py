from flask import Flask, request, jsonify
from model import EmbedModel
app = Flask(__name__)

model = EmbedModel()

@app.route("/get-embedding/<item_link>", methods=['GET'])
def get_embedding(item_link):
    try:
        embedding = model.embed(item_link)
        result = {
            "user_query": item_link,
            "embedding": embedding.tolist()
        }
        return jsonify(result), 200
    except Exception as e:
        print(e)
    return jsonify({}), 500 