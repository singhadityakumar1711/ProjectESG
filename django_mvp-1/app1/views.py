import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings


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

        # Return the response
        return JsonResponse(
            {
                "pdf_id": pdf_id,
                "file_url": file_url,
            }
        )

    return JsonResponse({"error": "Invalid request"}, status=400)
