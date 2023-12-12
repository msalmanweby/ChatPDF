from rest_framework import generics
from rest_framework.response import Response
from rest_framework import generics
from .ChatPDF import SimpleMethod, ComplexMethod

class QueryChain(generics.GenericAPIView):
    def post(self, request, *args, **kwargs):
        query = request.data.get('query')
        res = SimpleMethod(query)
        return Response({"answer": res})