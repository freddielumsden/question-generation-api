from rest_framework.response import Response
from rest_framework.decorators import api_view
# from generate_question.models import TextToQuestions
# from .serializers import TextToQuestionsSerializer
import generate_question.torch_model

@api_view(['GET'])
def getData(request):
    try:
        query_text = request.query_params['text']
    except:
        return Response({"error": "must supply query text"})
    #
    return Response(generate_question.torch_model.hf_run_model(query_text))