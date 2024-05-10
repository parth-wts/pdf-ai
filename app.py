from flask import Flask,request
from flask_restful import Resource, Api
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama


app = Flask(__name__)
api = Api(app)

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


class Chat(Resource):
    def post(self):
        # Process the data here
        data = request.get_json()  # Get JSON data from the request
        query_text = data.get('query');
        
        # Prepare the DB.
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_score(query_text, k=5)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        # print(prompt)

        model = Ollama(model="mistral")
        response_text = model.invoke(prompt)

        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        print(formatted_response)
    
        return {'received_data': formatted_response}; 201  # Return a response with status code 201 (Created)

api.add_resource(HelloWorld, '/')
api.add_resource(Chat, '/chat')

if __name__ == '__main__':
    app.run(debug=True)