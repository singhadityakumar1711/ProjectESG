import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine

# from pinecone import Pinecone, ServerlessSpec
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import download_loader
from llama_index.vector_stores.pinecone import PineconeVectorStore
import os
from getpass import getpass
import re
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_models import ChatOpenAI

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# from llama_index.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

# from llama_index.ingestion import IngestionPipeline
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec

PDFReader = download_loader("PDFReader")
loader = PDFReader()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
embed_model = OpenAIEmbedding(api_key=openai_api_key)


def clean_up_text(content: str) -> str:
    """
    Remove unwanted characters and patterns in text input.

    :param content: Text input.

    :return: Cleaned version of original text input.
    """

    # Fix hyphenated words broken by newline
    content = re.sub(r"(\w+)-\n(\w+)", r"\1\2", content)

    # Remove specific unwanted patterns and characters
    unwanted_patterns = [
        "\\n",
        "  —",
        "——————————",
        "—————————",
        "—————",
        r"\\u[\dA-Fa-f]{4}",
        r"\uf075",
        r"\uf0b7",
    ]
    for pattern in unwanted_patterns:
        content = re.sub(pattern, "", content)

    # Fix improperly spaced hyphenated words and normalize whitespace
    content = re.sub(r"(\w)\s*-\s*(\w)", r"\1-\2", content)
    content = re.sub(r"\s+", " ", content)

    return content


def res_parse(a):
    b = []
    for i in a.splitlines():
        b.append(i.replace("- ", ""))
    return b


prompts = [
    "Critically analyze the contribution in the field of Bribery & Corruption. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies eradicating Bribery & Corruption 2. Look thogroughly and strictly whether the company has identified the impacts of Bribery & Corruption on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Bribery & Corruption.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Training on principles. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies promoting Training on principles 2. Look thogroughly and strictly whether the company has identified the impacts of Training on principles on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Training on principles.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Transparency & disclosure in ethical, transparent manner. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies promoting Transparency & disclosure in ethical, transparent manner 2. Look thogroughly and strictly whether the company has identified the impacts of Transparency & disclosure in ethical, transparent manner on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Transparency & disclosure in ethical, transparent manner.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of R&D and capital expenditure. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies promoting R&D and capital expenditure 2. Look thogroughly and strictly whether the company has identified the impacts of R&D and capital expenditure on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to R&D and capital expenditure.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Input material and sourcing. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies promoting Input material and sourcing 2. Look thogroughly and strictly whether the company has identified the impacts of Input material and sourcing on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Input material and sourcing.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Usage of recycled or reused inputs. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies promoting Usage of recycled or reused inputs 2. Look thogroughly and strictly whether the company has identified the impacts of Usage of recycled or reused inputs on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Usage of recycled or reused inputs.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Well being of employees and workmen. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies promoting Well being of employees and workmen 2. Look thogroughly and strictly whether the company has identified the impacts of Well being of employees and workmen on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Well being of employees and workmen.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Details of trainings given to employees. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies towards Details of trainings given to employees 2. Look thogroughly and strictly whether the company has identified the impacts of Details of trainings given to employees on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Details of trainings given to employees.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Membership of employees in unions/ associations. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies promoting Membership of employees in unions/ associations 2. Look thogroughly and strictly whether the company has identified the impacts of Membership of employees in unions/ associations on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Membership of employees in unions/ associations.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Protection of stakeholder's interest. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies promoting Protection of stakeholder's interest 2. Look thogroughly and strictly whether the company has identified the impacts of Protection of stakeholder's interest on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Protection of stakeholder's interest.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Protection of Human rights. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies promoting Protection of Human rights 2. Look thogroughly and strictly whether the company has identified the impacts of protection of Human rights on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to protection of Human rights.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Details of salary/wages, minimum wages policy. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies promoting Details of salary/wages, minimum wages policy 2. Look thogroughly and strictly whether the company has identified the impacts of Details of salary/wages, minimum wages policy on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Details of salary/wages, minimum wages policy.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of grievances related to human rights issues such as child labor, sexual harassment etc. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies minimizing grievances related to human rights issues such as child labor, sexual harassment etc 2. Look thogroughly and strictly whether the company has identified the impacts of grievances related to human rights issues such as child labor, sexual harassment etc on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to grievances related to human rights issues such as child labor, sexual harassment etc.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Total Energy/ Electricity consumption. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies towards Total Energy/ Electricity consumption 2. Look thogroughly and strictly whether the company has identified the impacts of Total Energy/ Electricity consumption on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Total Energy/ Electricity consumption.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Water and Effluents disclosure. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies towards Water and Effluents disclosure 2. Look thogroughly and strictly whether the company has identified the impacts of Water and Effluents disclosure on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Water and Effluents disclosure.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of GHG Emissions. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies minimizing GHG Emissions 2. Look thogroughly and strictly whether the company has identified the impacts of GHG Emissions on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to GHG Emissions.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Waste, e-waste related disclosure. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies towards Waste, e-waste related disclosure 2. Look thogroughly and strictly whether the company has identified the impacts of Waste, e-waste related disclosure on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Waste, e-waste related disclosure.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Biodiversity. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies promoting Biodiversity 2. Look thogroughly and strictly whether the company has identified the impacts of Biodiversity on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to promoting Biodiversity.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Environmental impact assessments. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies promoting Environmental impact assessments 2. Look thogroughly and strictly whether the company has identified the impacts of Environmental impact assessments on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Environmental impact assessments.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Affiliation with association and industry chambers. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies promoting Affiliation with association and industry chambers 2. Look thogroughly and strictly whether the company has identified the impacts of Affiliation with association and industry chambers on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Affiliation with association and industry chambers.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of CSR beneficiaries. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies promoting CSR beneficiaries 2. Look thogroughly and strictly whether the company has identified the impacts of CSR beneficiaries on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to CSR beneficiaries.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Input material procurement. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies towards Input material procurement 2. Look thogroughly and strictly whether the company has identified the impacts of Input material procurement on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Input material procurement. Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Responding to consumer complaints and feedback. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies promoting Responding to consumer complaints and feedback 2. Look thogroughly and strictly whether the company has identified the impacts of Responding to consumer complaints and feedback on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Responding to consumer complaints and feedback.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Marketing and Labeling. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies promoting Marketing and Labeling 2. Look thogroughly and strictly whether the company has identified the impacts of Marketing and Labeling on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Marketing and Labeling.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
    "Critically analyze the contribution in the field of Customer Privacy. There are three metrics to analyze -1. Look thogroughly and strictly whether the company has alligned their policies promoting Customer Privacy 2. Look thogroughly and strictly whether the company has identified the impacts of Customer Privacy on operations 3. Look thogroughly and strictly whether the company has tracked the implementation and effectiveness of actions taken in response to Customer Privacy.Be strict when providing scores to the metrics. Just give me a number representing the count of the number of metrics that the company has addressed. No details are required. Just give me a value in digits",
]


def get_answer(query, index):
    """Fetches the answer to the query from the indexed PDF using LlamaIndex."""
    query_engine = index.as_query_engine()
    try:
        response = query_engine.query(query)
        if response:
            return response.response  # Assuming answer is in the "text" field
        else:
            return None
    except Exception as e:
        return None


def summarize_main(temp_dir):
    """Main function to handle user interaction and display results."""

    if temp_dir is not None:
        # Create a SimpleDirectoryReader instance
        # reader = SimpleDirectoryReader(temp_dir)

        # Load data from the reader
        # documents = reader.load_data()
        documents = loader.load_data(file=temp_dir)

        # Predefined queries (modify as needed)
        queries = [
            "Please make a very concise summarize in bullet points about the Water related sustainability initiatives like Water conservation, Rainwater harvesting, Rooftop rainwater harvesting, Wastewater treatment etc. Additionally, include water-related future goals and targets in the summary. Try to keep data and facts from tables as much as possible. Keep a character limit of 200 characters. Page number after each bullet point",
            "Please make a very concise summarize in bullet points about the Waste related sustainability initiatives like waste management, waste segregation, waste processing, waste disposal, e-waste management etc. Additionally include Waste related future goals and targets in the summary. Try to Keep data and facts from tables as much as possible. Keep a character limit of 200 characters. Page number after each bullet point",
            "Please make a very concise summarize in bullet points about Climate commitments, Scope1, Scope2 & Scope3 emission reduction strategies, Renewable energy, climate solutions, Carbon offset programs, Carbon neutral events etc. Additionally include all Climate related future goals and targets in the summary. Try to Keep data and facts from tables as much as possible. Keep a character limit of 200 characters. Page number after each bullet point",
            "Please make a very concise summarize in bullet points about Employee wellness initiatives like employee wellness training, Health and safety assessment, leadership development, Occupational Health and Safety measures, continuous performance management etc. Try to keep data and facts from tables as much as possible. Keep a character limit of 200 characters. Page number after each bullet point",
            "Please make a very concise summarize in bullet points about Diversity, Equity and Inclusion initiatives like managing employee attrition, enable employee performance, promotions, women in workforce, women in management, retaining employee with disabilities etc. Try to keep data and facts from tables as much as possible. Keep a character limit of 200 characters. Page number after each bullet point",
            "Please make a very concise summarize in bullet points about Human rights initiatives like employee human rights training, median salary structure, minimum wages details, human right complaints reporting mechanism etc. Try to keep data and facts from tables as much as possible. Keep a character limit of 200 characters. Give me information in bullet points. Page number after each bullet point",
            "Please make a very concise summarize in bullet points about Corporate Governance initiatives like building sustainable supply chains, ESG assessments for suppliers, Anti-Bribery and Anti-Corruption practices and policy,  Code of Conduct and Ethics, fair business practices, labor practices etc. Try to keep data and facts from tables as much as possible. Keep a character limit of 400 characters. Page number after each bullet point",
        ]
        index = VectorStoreIndex.from_documents(documents)
        responses = []

        for query in queries:
            responses.append(get_answer(query, index))

        summarized_response = {
            "Environmental": {
                "Water Sustainibilty": res_parse(responses[0]),
                "Waste Management": res_parse(responses[1]),
                "Climate Commitments": res_parse(responses[2]),
            },
            "Social": {
                "Employee wellness, Health & Safety": res_parse(responses[3]),
                "Diversity & Inclusion": res_parse(responses[4]),
                "Human Rights": res_parse(responses[5]),
            },
            "Governance": {
                "Corporate Governance": res_parse(responses[6]),
            },
        }

        summarized_response = json.dumps(summarized_response, indent=4)
        return summarized_response


@csrf_exempt
def ai_summarized(request):
    if request.method == "POST" and "pdf_id" in request.POST:
        pdf_id = request.POST["pdf_id"]
        mapping_file_path = os.path.join(
            settings.BASE_DIR, "public/app", "pdf_mappings.json"
        )
        with open(mapping_file_path, "r") as mapping_file:
            data = json.load(mapping_file)
            uploaded_file = data.get(pdf_id, None)
            return JsonResponse(summarize_main(uploaded_file), safe=False)


def assessment_main(temp_dir):
    if temp_dir is not None:

        documents = loader.load_data(file=temp_dir)
        index = VectorStoreIndex.from_documents(documents)
        loop = 0
        comment = []
        for prompt in prompts:
            score = get_answer(prompts[loop], index)
            if score == "1":
                comment.append("Not satisfied")
            elif score == "2":
                comment.append("Partially satisfied")
            else:
                comment.append("Completely satisfied")
            loop = loop + 1

        ai_checklist = [
            {
                "brsr": "Businesses should conduct and govern themselves with integrity and in a manner that is Ethical, Transparent and Accountable",
                "sdg": "SDG-16, SDG-17",
                "indicator_gri_and_assessment": [
                    {
                        "ind": "Bribery & Corruption",
                        "gri": "GRI 2-23, GRI 205-3",
                        "assessment": comment[0],
                    },
                    {
                        "ind": "Training on principles",
                        "gri": "GRI 2-17",
                        "assessment": comment[1],
                    },
                    {
                        "ind": "Transparency & disclosure",
                        "gri": "GRI 2-17",
                        "assessment": comment[2],
                    },
                ],
            },
            {
                "brsr": "Businesses should provide goods and services in a manner that is sustainable and safe",
                "sdg": "SDG-2, SDG-6, SDG-7, SDG-8, SDG-10, SDG-12, SDG-13, SDG-14, SDG-15",
                "indicator_gri_and_assessment": [
                    {
                        "ind": "R&D & capital expenditure",
                        "gri": "No direct linkage with GRI",
                        "assessment": comment[3],
                    },
                    {
                        "ind": "Input material and sourcing",
                        "gri": "GRI 308-1",
                        "assessment": comment[4],
                    },
                    {
                        "ind": "Usage of recycled or reused inputs",
                        "gri": "GRI 306-2",
                        "assessment": comment[5],
                    },
                ],
            },
            {
                "brsr": "Businesses should respect and promote the well-being of all employees, including, those in their value chains",
                "sdg": "SDG-1, SDG-3, SDG-4, SDG-5, SDG-8, SDG-9, SDG-11, SDG-16",
                "indicator_gri_and_assessment": [
                    {
                        "ind": "Well being of employees and workmen",
                        "gri": "GRI 401-2,GRI 401-3",
                        "assessment": comment[6],
                    },
                    {
                        "ind": "Details of trainings given to employees",
                        "gri": "GRI 404-1",
                        "assessment": comment[7],
                    },
                    {
                        "ind": "Membership of employees in unions/ associations",
                        "gri": "GRI 2-30",
                        "assessment": comment[8],
                    },
                ],
            },
            {
                "brsr": "Businesses should respect the interests of and be responsive to all its stakeholders",
                "sdg": "SDG-1, SDG-5, SDG-9, SDG-11, SDG-16",
                "indicator_gri_and_assessment": [
                    {
                        "ind": "Protection of stakeholder's interest",
                        "gri": "GRI 2-29",
                        "assessment": comment[9],
                    },
                ],
            },
            {
                "brsr": "Businesses should respect and promote human rights",
                "sdg": "SDG-5, SDG-8, SDG-16",
                "indicator_gri_and_assessment": [
                    {
                        "ind": "Protection of Human rights",
                        "gri": "GRI 2-24",
                        "assessment": comment[10],
                    },
                    {
                        "ind": "Details of salary/wages, minimum wages policy",
                        "gri": "GRI 2-19, GRI 2-21, GRI 405-2",
                        "assessment": comment[11],
                    },
                    {
                        "ind": "Grievances related to human rights issues such as child labor, sexual harassment etc",
                        "gri": "GRI 2-13, GRI 2-25, GRI 406-1",
                        "assessment": comment[12],
                    },
                ],
            },
            {
                "brsr": "Businesses should respect and make efforts to protect and restore the environment",
                "sdg": "SDG-2, SDG-3, SDG-6, SDG-7, SDG-10, SDG-12, SDG-13, SDG-14, SDG-15",
                "indicator_gri_and_assessment": [
                    {
                        "ind": "Total Energy/ Electricity consumption",
                        "gri": "GRI 302-1, GRI 302-3",
                        "assessment": comment[13],
                    },
                    {
                        "ind": "Water and Effluents disclosure",
                        "gri": "GRI 303-1, GRI 303-2, GRI 303-3, GRI 303-4, GRI 303-5",
                        "assessment": comment[14],
                    },
                    {
                        "ind": "GHG Emissions",
                        "gri": "GRI 305-1, GRI 305-2, GRI 305-4, GRI 305-5",
                        "assessment": comment[15],
                    },
                    {
                        "ind": "Waste, e-waste related disclosure",
                        "gri": "GRI 306-3, GRI 306-4, GRI 306-5, GRI 306-2",
                        "assessment": comment[16],
                    },
                    {
                        "ind": "Biodiversity",
                        "gri": "GRI 304-1",
                        "assessment": comment[17],
                    },
                    {
                        "ind": "Environmental impact assessments",
                        "gri": "GRI 413-1, GRI 303-1",
                        "assessment": comment[18],
                    },
                ],
            },
            {
                "brsr": "Businesses, when engaging in influencing public and regulatory policy, should do so in a manner that is responsible and transparent",
                "sdg": "SDG-2, SDG-7, SDG-10, SDG-11, SDG-13, SDG-14, SDG-15, SDG-17",
                "indicator_gri_and_assessment": [
                    {
                        "ind": "Affiliation with association & industry chambers",
                        "gri": "GRI 2-28",
                        "assessment": comment[19],
                    },
                ],
            },
            {
                "brsr": "Businesses should promote inclusive growth and equitable development",
                "sdg": "SDG-1, SDG-2, SDG-3, SDG-5, SDG-6, SDG-8, SDG-9, SDG-11, SDG-(13-17)",
                "indicator_gri_and_assessment": [
                    {
                        "ind": "CSR beneficiaries",
                        "gri": "GRI 413-1, GRI 2-25",
                        "assessment": comment[20],
                    },
                    {
                        "ind": "Input material procurement",
                        "gri": "GRI 204-1",
                        "assessment": comment[21],
                    },
                ],
            },
            {
                "brsr": "Businesses should engage with and provide value to their consumers in a responsible manner",
                "sdg": "SDG-2, SDG-4, SDG-12, SDG-14, SDG-15",
                "indicator_gri_and_assessment": [
                    {
                        "ind": "Responding to consumer complaints and feedback",
                        "gri": "GRI 2-25",
                        "assessment": comment[22],
                    },
                    {
                        "ind": "Marketing and Labeling",
                        "gri": "GRI 417",
                        "assessment": comment[23],
                    },
                    {
                        "ind": "Customer Privacy",
                        "gri": "GRI 418",
                        "assessment": comment[24],
                    },
                ],
            },
        ]

        ai_checklist = json.dumps(ai_checklist, indent=4)
        return ai_checklist


@csrf_exempt
def ai_principle_checklist(request):
    if request.method == "POST" and "pdf_id" in request.POST:
        pdf_id = request.POST["pdf_id"]
        mapping_file_path = os.path.join(
            settings.BASE_DIR, "public/app", "pdf_mappings.json"
        )
        with open(mapping_file_path, "r") as mapping_file:
            data = json.load(mapping_file)
            uploaded_file = data.get(pdf_id, None)
            return JsonResponse(assessment_main(uploaded_file), safe=False)


@csrf_exempt  # Disable CSRF protection for this view (not recommended for production)
def upload_pdf(request):
    if (
        request.method == "POST"
        and "pdf_file" in request.FILES
        and "pdf_id" in request.POST
    ):
        pdf_file = request.FILES["pdf_file"]
        pdf_id = request.POST["pdf_id"]

        # Define the path to the public directory
        public_dir = os.path.join(settings.BASE_DIR, "public/app")

        # Create the directory if it doesn't exist
        if not os.path.exists(public_dir):
            os.makedirs(public_dir)

        # Define the file path
        file_path = os.path.join(public_dir, f"{pdf_id}.pdf")

        # Save the file
        with open(file_path, "wb+") as destination:
            for chunk in pdf_file.chunks():
                destination.write(chunk)

        temp = f"{pdf_id}.pdf"
        # Construct the URL
        file_url = request.build_absolute_uri(f"/public/app/{temp}")

        mapping_file_path = os.path.join(
            settings.BASE_DIR, "public/app", "pdf_mappings.json"
        )
        if os.path.exists(mapping_file_path):
            with open(mapping_file_path, "r") as mapping_file:
                pdf_mappings = json.load(mapping_file)
        else:
            pdf_mappings = {}

        pdf_mappings[pdf_id] = file_path

        with open(mapping_file_path, "w") as mapping_file:
            json.dump(pdf_mappings, mapping_file, indent=4)

        # Return the response
        return JsonResponse(
            {
                "pdf_id": pdf_id,
                "file_url": file_url,
            }
        )

    return JsonResponse({"error": "Invalid request"}, status=400)
    
def namespace_exists(namespace, index):
    index_stats = index.describeIndexStats()
    return namespace in index_stats["namespaces"]

@csrf_exempt
def ai_chat_load(request):
    if request.method == "POST":
        pdf_ids = request.POST.get("pdf_ids", "[]")
        pdf_ids = json.loads(pdf_ids)
        if not isinstance(pdf_ids, list):
            return JsonResponse(
                {"error": "Invalid data format. 'pdf_ids' should be a list."},
                status=400,
            )

        temp_index = ""

        mapping_file_path = os.path.join(
            settings.BASE_DIR, "public/app", "pdf_mappings.json"
        )
        with open(mapping_file_path, "r") as mapping_file:
            data = json.load(mapping_file)
            doc_list = []

            for id in pdf_ids:
                temp_index = temp_index + id
                uploaded_file = data.get(id)
                print(uploaded_file)
                document = loader.load_data(file=uploaded_file)
                doc_list.extend(document)

            pc = PineconeGRPC(api_key=pinecone_api_key)
            index_name = "esg-genai"
            pinecone_index = pc.Index(index_name)
            if(namespace_exists(temp_index, pinecone_index)):
                return JsonResponse({"Docs_index": temp_index})
                
            # print(len(doc_list))
            # print(doc_list)
            cleaned_docs = []
            for d in doc_list:
                cleaned_text = clean_up_text(d.text)
                d.text = cleaned_text
                cleaned_docs.append(d)

            

            # Create your index (can skip this step if your index already exists)
            # pc.create_index(
            #     index_name,
            #     dimension=1536,
            #     spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            # )

            # Initialize your index
            

            # Initialize VectorStore
            vector_store = PineconeVectorStore(
                pinecone_index=pinecone_index, namespace=temp_index
            )
            pipeline = IngestionPipeline(
                transformations=[
                    SentenceSplitter(chunk_size=1000000, chunk_overlap=0),
                    TitleExtractor(),
                    OpenAIEmbedding(),
                ],
                vector_store=vector_store,
            )
            pipeline.run(documents=cleaned_docs)
            return JsonResponse({"Docs_index": temp_index})



@csrf_exempt
def ai_chat_query(request):
    if (
        request.method == "POST"
        and "combined_id" in request.POST
        and "query" in request.POST
    ):
        temp_index = request.POST["combined_id"]
        query = request.POST["query"]
        pc = PineconeGRPC(api_key=pinecone_api_key)
        index_name = "esg-genai"
        pinecone_index = pc.Index(index_name)
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = openai_api_key

            # Instantiate VectorStoreIndex object from our vector_store object
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index, namespace=temp_index
        )
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=1)
        query_engine = RetrieverQueryEngine(retriever=retriever)
        llm_query = query_engine.query(query)
        # index = vector_dict[temp_index]
        return JsonResponse({"response": llm_query.response})


def user_input(user_question, tools):
    prompt = hub.pull("hwchase17/openai-functions-agent")
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo-1106",
        api_key=openai_api_key,
        request_timeout=180,
    )

    prompt2 = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant for question-answering tasks, who also is an expert at routing a user question to a vectorstore or web search. Use the vectorstore for questions on sustainibility reporting, ESG disclosures, SEBI BRSR guidelines etc . However you do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search. The goal is to filter out erroneous retrievals. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. You have access to the following tools:{tools}",
            ),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")

    llm2 = "llama3"
    agent = create_openai_tools_agent(llm, tools, prompt2)

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    response = agent_executor.invoke({"input": user_question})
    return response["output"]


# def get_vector_store():
#         urls = [
#             "https://www.sebi.gov.in/sebi_data/commondocs/may-2021/Business%20responsibility%20and%20sustainability%20reporting%20by%20listed%20entitiesAnnexure1_p.PDF",
#             "https://www.sebi.gov.in/legal/circulars/jul-2023/brsr-core-framework-for-assurance-and-esg-disclosures-for-value-chain_73854.html",
#             #  "https://www.sebi.gov.in/sebi_data/commondocs/may-2021/Business%20responsibility%20and%20sustainability%20reporting%20by%20listed%20entitiesAnnexure2_p.PDF",
#             #  "https://www.sebi.gov.in/sebi_data/commondocs/jul-2023/Annexure_II-Updated-BRSR_p.PDF",
#             #"https://www.nseindia.com/companies-listing/corporate-filings-bussiness-sustainabilitiy-reports"
#             ]
#         docs = [WebBaseLoader(url).load() for url in urls]
#         print(docs[0])
#         # print(docs[:3])
#         docs_list = [item for sublist in docs for item in sublist]
#         print("-----------")
#         # print(docs_list[:3])
#         documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=10).split_documents(docs_list)
#         # print(documents)
#         # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#         pc = PineconeGRPC(api_key=pinecone_api_key)
#         index_name = "esg-genai"
#         pinecone_index = pc.Index(index_name)
#         vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace = "web_agent")
#         pipeline = IngestionPipeline(
#                     transformations=[
#                         SentenceSplitter(chunk_size=10000, chunk_overlap=0),
#                         TitleExtractor(),
#                         OpenAIEmbedding(),
#                     ],
#                     vector_store=vector_store,
#                 )
#         pipeline.run(documents=documents)
#         #  vector_store=FAISS.from_documents(documents,embedding=embeddings)
#         #  vector_store.save_local("brsr_faiss_index")
#          #return vectordb

# @csrf_exempt
# def ai_agent(request):
#     if (
#         request.method == "POST"
#         and "query" in request.POST
#     ):
#         query = request.POST["query"]
#         tavily_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_api_key)
#         tavily = TavilySearchResults(api_wrapper=tavily_wrapper)
#         get_vector_store()
#         # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#         # new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#         # retriever=new_db.as_retriever()
#         pc = PineconeGRPC(api_key=pinecone_api_key)
#         index_name = "esg-genai"
#         pinecone_index = pc.Index(index_name)
#         if not os.getenv("OPENAI_API_KEY"):
#             os.environ["OPENAI_API_KEY"] = openai_api_key

#                 # Instantiate VectorStoreIndex object from our vector_store object
#         vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace="web_agent")
#         vector_index = VectorStoreIndex.from_vector_store(
#             vector_store=vector_store
#         )
#         retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=1)
#         retriever_tool=create_retriever_tool(retriever,"brsr_search",
#                                             "Search for information about brsr & sustainibility. For any questions about india sustainibility & BRSR, you must use this tool!"
#                                             )


#         api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
#         wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

#         tools=[retriever_tool,tavily,wiki]
#         response = user_input(query, tools)
#         return JsonResponse({"response": response})


def get_vector_store():
    #  Append your URLs to search in the list below
    urls = [
        "https://www.sebi.gov.in/sebi_data/commondocs/may-2021/Business%20responsibility%20and%20sustainability%20reporting%20by%20listed%20entitiesAnnexure1_p.PDF",
        "https://www.sebi.gov.in/legal/circulars/jul-2023/brsr-core-framework-for-assurance-and-esg-disclosures-for-value-chain_73854.html",
        #  "https://www.sebi.gov.in/sebi_data/commondocs/may-2021/Business%20responsibility%20and%20sustainability%20reporting%20by%20listed%20entitiesAnnexure2_p.PDF",
        #  "https://www.sebi.gov.in/sebi_data/commondocs/jul-2023/Annexure_II-Updated-BRSR_p.PDF",
        # "https://www.nseindia.com/companies-listing/corporate-filings-bussiness-sustainabilitiy-reports"
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    documents = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=10
    ).split_documents(docs_list)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")
    # return vectordb


@csrf_exempt
def ai_agent(request):
    if request.method == "POST" and "query" in request.POST:
        query = request.POST["query"]
        # Uncomment and run the query once to generate the vectors after appending new URLs
        # get_vector_store()
        tavily_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_api_key)
        tavily = TavilySearchResults(api_wrapper=tavily_wrapper)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        new_db = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
        retriever = new_db.as_retriever()
        retriever_tool = create_retriever_tool(
            retriever,
            "brsr_search",
            "Search for information about brsr & sustainibility. For any questions about india sustainibility & BRSR, you must use this tool!",
        )

        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

        tools = [retriever_tool, tavily, wiki]
        response = user_input(query, tools)
        return JsonResponse({"response": response})
