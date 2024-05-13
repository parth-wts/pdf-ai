from flask import Flask, request
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from get_embedding_function import get_embedding_function

app = Flask(__name__)

folder_path = "db"

cached_llm = Ollama(model="mistral")

# Prepare the DB.
embedding_function = get_embedding_function()

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching docuemnts. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

MONGODB_ATLAS_CLUSTER_URI = "mongodb+srv://naitikjoshi:Niletree%2323@vector-store.eqaw6.mongodb.net/?retryWrites=true&w=majority&appName=vector-store"

# initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)


DB_NAME = "vstore"
COLLECTION_NAME = "vectors"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response = cached_llm.invoke(query)

    print(response)

    response_answer = {"answer": response}
    return response_answer


@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")
    
    
    # vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        MONGODB_ATLAS_CLUSTER_URI,
        DB_NAME + "." + COLLECTION_NAME,
        embedding=embedding_function,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
    
    retriever =  vector_search.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 25,
            "score_threshold":0.4
        },
    )
    
    print("*********************")
    print(retriever)
    print("*********************")


    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer


@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    # vector_store = Chroma.from_documents(
    #     documents=chunks, embedding=embedding, persist_directory=folder_path
    # )

    # vector_store.persist()


    # insert the documents in MongoDB Atlas with their embedding
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=embedding_function,
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()
