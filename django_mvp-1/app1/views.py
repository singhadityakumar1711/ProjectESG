import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

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
        public_dir = os.path.join(settings.BASE_DIR, "public")

        # Create the directory if it doesn't exist
        if not os.path.exists(public_dir):
            os.makedirs(public_dir)

        # Define the file path
        file_path = os.path.join(public_dir, pdf_file.name)

        # Save the file
        with open(file_path, "wb+") as destination:
            for chunk in pdf_file.chunks():
                destination.write(chunk)

        # Construct the URL
        file_url = request.build_absolute_uri(f"/public/{pdf_file.name}")

        mapping_file_path = os.path.join(settings.BASE_DIR, 'public', 'pdf_mappings.json')
        if os.path.exists(mapping_file_path):
            with open(mapping_file_path, 'r') as mapping_file:
                pdf_mappings = json.load(mapping_file)
        else:
            pdf_mappings = {}

        pdf_mappings[pdf_id] = file_path

        with open(mapping_file_path, 'w') as mapping_file:
            json.dump(pdf_mappings, mapping_file, indent=4)

        # Return the response
        return JsonResponse(
            {
                "pdf_id": pdf_id,
                "file_url": file_url,
            }
        )

    return JsonResponse({"error": "Invalid request"}, status=400)


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
            reader = SimpleDirectoryReader(temp_dir)

            # Load data from the reader
            documents = reader.load_data()

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
                "Environmental":{
                    "Water Sustainibilty":responses[0],
                    "Waste Management": responses[1],
                    "Climate Commitments": responses[2],
                },
                "Social":{
                    "Employee wellness, Health & Safety": responses[3],
                    "Diversity & Inclusion": responses[4],
                    "Human Rights": responses[5],
                },
                "Governance":{
                    "Corporate Governance": responses[6],
                },
            }

            summarized_response = json.dumps(summarized_response, indent=4)
            return summarized_response
            


@csrf_exempt
def ai_summarized(request):
    if (
        request.method == "POST"
        and "pdf_id" in request.POST
    ):
        pdf_id = request.POST["pdf_id"]
        mapping_file_path = os.path.join(settings.BASE_DIR, 'public', 'pdf_mappings.json')
        with open(mapping_file_path, 'r') as mapping_file:
            data = json.load(mapping_file)
            uploaded_file = data.get(pdf_id, None)
            return JsonResponse(summarize_main(os.path.dirname(uploaded_file)), safe=False)
            
        
