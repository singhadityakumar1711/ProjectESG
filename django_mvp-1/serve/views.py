from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .summarize import generate_response  # Import your LLM function

class QueryView(APIView):
    def get(self, request, format=None):
        # query = "Give me a short summary of the document"
        response_text = generate_response()
        print(response_text)  # Print the response on the server screen
        return Response({"response": response_text}, status=status.HTTP_200_OK)
