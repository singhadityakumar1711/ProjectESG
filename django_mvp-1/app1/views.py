import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


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


@csrf_exempt  # Disable CSRF protection for this view (not recommended for production)
def upload_pdf_app1(request):
    if (
        request.method == "POST"
        and "pdf_file" in request.FILES
        and "pdf_id" in request.POST
    ):
        pdf_file = request.FILES["pdf_file"]
        pdf_id = request.POST["pdf_id"]

        # Define the path to the public directory
        public_dir = os.path.join(settings.BASE_DIR, "public\\app1")

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
        file_url = request.build_absolute_uri(f"/public/app1/{pdf_file.name}")

        mapping_file_path = os.path.join(
            settings.BASE_DIR, "public\\app1", "pdf_mappings_app1.json"
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
            "Environmental": {
                "Water Sustainibilty": responses[0],
                "Waste Management": responses[1],
                "Climate Commitments": responses[2],
            },
            "Social": {
                "Employee wellness, Health & Safety": responses[3],
                "Diversity & Inclusion": responses[4],
                "Human Rights": responses[5],
            },
            "Governance": {
                "Corporate Governance": responses[6],
            },
        }

        summarized_response = json.dumps(summarized_response, indent=4)
        return summarized_response


@csrf_exempt
def ai_summarized(request):
    if request.method == "POST" and "pdf_id" in request.POST:
        pdf_id = request.POST["pdf_id"]
        mapping_file_path = os.path.join(
            settings.BASE_DIR, "public\\app1", "pdf_mappings_app1.json"
        )
        with open(mapping_file_path, "r") as mapping_file:
            data = json.load(mapping_file)
            uploaded_file = data.get(pdf_id, None)
            return JsonResponse(
                summarize_main(os.path.dirname(uploaded_file)), safe=False
            )


@csrf_exempt  # Disable CSRF protection for this view (not recommended for production)
def upload_pdf_app2(request):
    if (
        request.method == "POST"
        and "pdf_file" in request.FILES
        and "pdf_id" in request.POST
    ):
        pdf_file = request.FILES["pdf_file"]
        pdf_id = request.POST["pdf_id"]

        # Define the path to the public directory
        public_dir = os.path.join(settings.BASE_DIR, "public\\app2")

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
        file_url = request.build_absolute_uri(f"/public/app2/{pdf_file.name}")

        mapping_file_path = os.path.join(
            settings.BASE_DIR, "public\\app2", "pdf_mappings_app2.json"
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



def assessment_main(temp_dir):
    # st.title("ESG Report Assessment")
    # st.write("**Upload BRSR Document (PDF format only)**")
    # uploaded_file_1 = st.file_uploader("Choose a PDF:", type="pdf")
    if temp_dir is not None:
        # Create a temporary directory for indexing
        # st.header("**BRSR Principles Checklist**")
        # with tempfile.TemporaryDirectory() as temp_dir_1:
            # Save uploaded file
            # pdf_path = f"{temp_dir_1}/{uploaded_file_1.name}"
            # with open(pdf_path, "wb") as f:
            #     f.write(uploaded_file_1.read())

            # Create a SimpleDirectoryReader instance
            reader = SimpleDirectoryReader(temp_dir)

            # Load data from the reader
            documents = reader.load_data()
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
                    "brsr":"Businesses should conduct and govern themselves with integrity and in a manner that is Ethical, Transparent and Accountable",
                    "sdg":"SDG-16, SDG-17",
                    "indicator_gri_and_assessment":[
                        {"ind":"Bribery & Corruption", "gri":"GRI 2-23, GRI 205-3", "assessment":comment[0]},
                        {"ind":"Training on principles", "gri":"GRI 2-17", "assessment":comment[1]},
                        {"ind":"Transparency & disclosure", "gri":"GRI 2-17", "assessment":comment[2]},
                    ],
                },
                {
                    "brsr":"Businesses should provide goods and services in a manner that is sustainable and safe",
                    "sdg":"SDG-2, SDG-6, SDG-7, SDG-8, SDG-10, SDG-12, SDG-13, SDG-14, SDG-15",
                    "indicator_gri_and_assessment":[
                        {"ind":"R&D & capital expenditure", "gri":"No direct linkage with GRI", "assessment":comment[3]},
                        {"ind":"Input material and sourcing", "gri":"GRI 308-1", "assessment":comment[4]},
                        {"ind":"Usage of recycled or reused inputs", "gri":"GRI 306-2", "assessment":comment[5]},
                    ],
                },
                {
                    "brsr":"Businesses should respect and promote the well-being of all employees, including, those in their value chains",
                    "sdg":"SDG-1, SDG-3, SDG-4, SDG-5, SDG-8, SDG-9, SDG-11, SDG-16",
                    "indicator_gri_and_assessment":[
                        {"ind":"Well being of employees and workmen", "gri":"GRI 401-2,GRI 401-3", "assessment":comment[6]},
                        {"ind":"Details of trainings given to employees", "gri":"GRI 404-1", "assessment":comment[7]},
                        {"ind":"Membership of employees in unions/ associations", "gri":"GRI 2-30", "assessment":comment[8]},
                    ],
                },
                {
                    "brsr":"Businesses should respect the interests of and be responsive to all its stakeholders",
                    "sdg":"SDG-1, SDG-5, SDG-9, SDG-11, SDG-16",
                    "indicator_gri_and_assessment":[
                        {"ind":"Protection of stakeholder's interest", "gri":"GRI 2-29", "assessment":comment[9]},
                    ],
                },
                {
                    "brsr":"Businesses should respect and promote human rights",
                    "sdg":"SDG-5, SDG-8, SDG-16",
                    "indicator_gri_and_assessment":[
                        {"ind":"Protection of Human rights", "gri":"GRI 2-24", "assessment":comment[10]},
                        {"ind":"Details of salary/wages, minimum wages policy", "gri":"GRI 2-19, GRI 2-21, GRI 405-2", "assessment":comment[11]},
                        {"ind":"Grievances related to human rights issues such as child labor, sexual harassment etc", "gri":"GRI 2-13, GRI 2-25, GRI 406-1", "assessment":comment[12]},
                    ],
                },
                {
                    "brsr":"Businesses should respect and make efforts to protect and restore the environment",
                    "sdg":"SDG-2, SDG-3, SDG-6, SDG-7, SDG-10, SDG-12, SDG-13, SDG-14, SDG-15",
                    "indicator_gri_and_assessment":[
                        {"ind":"Total Energy/ Electricity consumption", "gri":"GRI 302-1, GRI 302-3", "assessment":comment[13]},
                        {"ind":"Water and Effluents disclosure", "gri":"GRI 303-1, GRI 303-2, GRI 303-3, GRI 303-4, GRI 303-5", "assessment":comment[14]},
                        {"ind":"GHG Emissions", "gri":"GRI 305-1, GRI 305-2, GRI 305-4, GRI 305-5", "assessment":comment[15]},
                        {"ind":"Waste, e-waste related disclosure", "gri":"GRI 306-3, GRI 306-4, GRI 306-5, GRI 306-2", "assessment":comment[16]},
                        {"ind":"Biodiversity", "gri":"GRI 304-1", "assessment":comment[17]},
                        {"ind":"Environmental impact assessments", "gri":"GRI 413-1, GRI 303-1", "assessment":comment[18]},
                    ],
                },
                {
                    "brsr":"Businesses, when engaging in influencing public and regulatory policy, should do so in a manner that is responsible and transparent",
                    "sdg":"SDG-2, SDG-7, SDG-10, SDG-11, SDG-13, SDG-14, SDG-15, SDG-17",
                    "indicator_gri_and_assessment":[
                        {"ind":"Affiliation with association & industry chambers", "gri":"GRI 2-28", "assessment":comment[19]},
                    ],
                },
                {
                    "brsr":"Businesses should promote inclusive growth and equitable development",
                    "sdg":"SDG-1, SDG-2, SDG-3, SDG-5, SDG-6, SDG-8, SDG-9, SDG-11, SDG-(13-17)",
                    "indicator_gri_and_assessment":[
                        {"ind":"CSR beneficiaries", "gri":"GRI 413-1, GRI 2-25", "assessment":comment[20]},
                        {"ind":"Input material procurement", "gri":"GRI 204-1", "assessment":comment[21]},
                    ],
                },
                {
                    "brsr":"Businesses should engage with and provide value to their consumers in a responsible manner",
                    "sdg":"SDG-2, SDG-4, SDG-12, SDG-14, SDG-15",
                    "indicator_gri_and_assessment":[
                        {"ind":"Responding to consumer complaints and feedback", "gri":"GRI 2-25", "assessment":comment[22]},
                        {"ind":"Marketing and Labeling", "gri":"GRI 417", "assessment":comment[23]},
                        {"ind":"Customer Privacy", "gri":"GRI 418", "assessment":comment[24]},
                    ],
                },
            ]

            ai_checklist = json.dumps(ai_checklist, indent=4)
            return ai_checklist

            # data["Assessment"] = comment
            # df = pd.DataFrame(data)

          #   html = """
          # <table border="1" style="margin: 0 auto; width:100%; border-collapse: collapse; text-align: center;">
          # <thead>
          #   <tr>
          #     <th>Principle#</th>
          #     <th>BRSR Principles</th>
          #     <th>SDG Goals</th>
          #     <th>Indicators</th>
          #     <th>GRI Mapping</th>
          #     <th>Assesssment</th>
          #   </tr>
          # </thead>
          # <tbody>
          #   <tr>
          #     <td rowspan="3"><strong>Principle 1</td>
          #     <td rowspan="3">Businesses should conduct and govern themselves with integrity and in a manner that is Ethical, Transparent and Accountable</td>
          #     <td rowspan="3">SDG-16, SDG-17</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   </tr>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   </tr>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   <tr>
          #     <td rowspan="3"><strong>Principle 2</td>
          #     <td rowspan="3">Businesses should provide goods and services in a manner that is sustainable and safe</td>
          #     <td rowspan="3">SDG-2, SDG-6, SDG-7, SDG-8, SDG-10, SDG-12, SDG-13, SDG-14, SDG-15</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   </tr>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   </tr>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   <tr>
          #     <td rowspan="3"><strong>Principle 3</td>
          #     <td rowspan="3">Businesses should respect and promote the well-being of all employees, including, those in their value chains</td>
          #     <td rowspan="3">SDG-1, SDG-3, SDG-4, SDG-5, SDG-8, SDG-9, SDG-11, SDG-16</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   </tr>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   </tr>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   <tr>
          #     <td><strong>Principle 4</td>
          #     <td>Businesses should respect the interests of and be responsive to all its stakeholders</td>
          #     <td>SDG-1, SDG-5, SDG-9, SDG-11, SDG-16</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   <tr>
          #     <td rowspan="3"><strong>Principle 5</td>
          #     <td rowspan="3">Businesses should respect and promote human rights</td>
          #     <td rowspan="3">SDG-5, SDG-8, SDG-16</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   </tr>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   </tr>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   <tr>
          #     <td rowspan="6"><strong>Principle 6</td>
          #     <td rowspan="6">Businesses should respect and make efforts to protect and restore the environment</td>
          #     <td rowspan="6">SDG-2, SDG-3, SDG-6, SDG-7, SDG-10, SDG-12, SDG-13, SDG-14, SDG-15</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   </tr>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   </tr>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   </tr>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   </tr>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   </tr>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   <tr>
          #     <td><strong>Principle 7</td>
          #     <td>Businesses, when engaging in influencing public and regulatory policy, should do so in a manner that is responsible and transparent</td>
          #     <td>SDG-2, SDG-7, SDG-10, SDG-11, SDG-13, SDG-14, SDG-15, SDG-17</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   <tr>
          #     <td rowspan="2"><strong>Principle 8</td>
          #     <td rowspan="2">Businesses should promote inclusive growth and equitable development</td>
          #     <td rowspan="2">SDG-1, SDG-2, SDG-3, SDG-5, SDG-6, SDG-8, SDG-9, SDG-11, SDG-(13-17)</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   </tr>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   <tr>
          #     <td rowspan="3"><strong>Principle 9</td>
          #     <td rowspan="3">Businesses should engage with and provide value to their consumers in a responsible manner</td>
          #     <td rowspan="3">SDG-2, SDG-4, SDG-12, SDG-14, SDG-15</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   </tr>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          #   </tr>
          #     <td>{}</td>
          #     <td>{}</td>
          #     <td>{}</td>
          #   </tr>
          # </tbody>
          # </table>
          # """.format(
          #       df.iloc[1, 3],
          #       df.iloc[1, 4],
          #       df.iloc[1, 5],
          #       df.iloc[2, 3],
          #       df.iloc[2, 4],
          #       df.iloc[2, 5],
          #       df.iloc[3, 3],
          #       df.iloc[3, 4],
          #       df.iloc[3, 5],
          #       df.iloc[4, 3],
          #       df.iloc[4, 4],
          #       df.iloc[4, 5],
          #       df.iloc[5, 3],
          #       df.iloc[5, 4],
          #       df.iloc[5, 5],
          #       df.iloc[6, 3],
          #       df.iloc[6, 4],
          #       df.iloc[6, 5],
          #       df.iloc[7, 3],
          #       df.iloc[7, 4],
          #       df.iloc[7, 5],
          #       df.iloc[8, 3],
          #       df.iloc[8, 4],
          #       df.iloc[8, 5],
          #       df.iloc[9, 3],
          #       df.iloc[9, 4],
          #       df.iloc[9, 5],
          #       df.iloc[10, 3],
          #       df.iloc[10, 4],
          #       df.iloc[10, 5],
          #       df.iloc[11, 3],
          #       df.iloc[11, 4],
          #       df.iloc[11, 5],
          #       df.iloc[12, 3],
          #       df.iloc[12, 4],
          #       df.iloc[12, 5],
          #       df.iloc[13, 3],
          #       df.iloc[13, 4],
          #       df.iloc[13, 5],
          #       df.iloc[14, 3],
          #       df.iloc[14, 4],
          #       df.iloc[14, 5],
          #       df.iloc[15, 3],
          #       df.iloc[15, 4],
          #       df.iloc[15, 5],
          #       df.iloc[16, 3],
          #       df.iloc[16, 4],
          #       df.iloc[16, 5],
          #       df.iloc[17, 3],
          #       df.iloc[17, 4],
          #       df.iloc[17, 5],
          #       df.iloc[18, 3],
          #       df.iloc[18, 4],
          #       df.iloc[18, 5],
          #       df.iloc[19, 3],
          #       df.iloc[19, 4],
          #       df.iloc[19, 5],
          #       df.iloc[20, 3],
          #       df.iloc[20, 4],
          #       df.iloc[20, 5],
          #       df.iloc[21, 3],
          #       df.iloc[21, 4],
          #       df.iloc[21, 5],
          #       df.iloc[22, 3],
          #       df.iloc[22, 4],
          #       df.iloc[22, 5],
          #       df.iloc[23, 3],
          #       df.iloc[23, 4],
          #       df.iloc[23, 5],
          #       df.iloc[24, 3],
          #       df.iloc[24, 4],
          #       df.iloc[24, 5],
          #       df.iloc[25, 3],
          #       df.iloc[25, 4],
          #       df.iloc[25, 5],
          #   )

            # Display the custom HTML in Streamlit
            # st.markdown(html, unsafe_allow_html=True)


@csrf_exempt
def ai_principle_checklist(request):
    if request.method == "POST" and "pdf_id" in request.POST:
        pdf_id = request.POST["pdf_id"]
        mapping_file_path = os.path.join(
            settings.BASE_DIR, "public\\app2", "pdf_mappings_app2.json"
        )
        with open(mapping_file_path, "r") as mapping_file:
            data = json.load(mapping_file)
            uploaded_file = data.get(pdf_id, None)
            return JsonResponse(
                assessment_main(os.path.dirname(uploaded_file)), safe=False
            )