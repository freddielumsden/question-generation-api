from rest_framework import serializers
from generate_question.models import TextToQuestions


class TextToQuestionsSerializer(serializers.ModelSerializer):
    class Meta:
        model = TextToQuestions
        fields = "__all__"
