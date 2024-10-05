import os
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv

# ---------Uncomment the below portion of the code to access your API-KEY from the .env folder------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def generate_response():
    documents = SimpleDirectoryReader(
        input_files=["C:/Users/ASUS/Desktop/GreenR/tcs.pdf"]
    ).load_data()
    # print(documents[:5])
    # Create the index
    index = VectorStoreIndex.from_documents(documents)

    # Create a query engine
    query_engine = index.as_query_engine()

    # Query the engine
    response = query_engine.query(
        "What is TCS doing to curtail it's GHG Emissions.Please elaborate in bullet points"
    )

    # Print the response
    return f"Response to: {response.response}"
