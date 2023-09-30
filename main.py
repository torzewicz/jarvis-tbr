import os
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain import PromptTemplate, LLMChain
from elevenlabs import generate, play, set_api_key, voices
import speech_recognition as sr

r = sr.Recognizer()

os.environ["OPENAI_API_KEY"] = 'OPENAI_KEY'
set_api_key('ELVENLABS_KEY')

# For chat gpt 3.5
llm = ChatOpenAI(model_name='gpt-3.5-turbo')

# For chat gpt 4
# llm = AzureChatOpenAI(
#     openai_api_base=API_BASE_AZURE,
#     openai_api_version="2023-05-15",
#     deployment_name=DEPLOYMENT_NAME_AZURE,
#     openai_api_key=API_KEY_AZURE,
#     openai_api_type="azure",
# )

prompt_template = "Jesteś pomocniczym asystentem AI o nazwie Jarvis. Odpowiadasz na pytania. Pytanie: {context}"

voices = voices()

qa = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))

while True:
    try:
        print('Speak')
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration=0.2)
            audio2 = r.listen(source2)
            recognized_text = r.recognize_google(audio2, language="pl-PL")
            print("Question: " + recognized_text)
            llm_response = qa(recognized_text)
            response = llm_response['text']
            print("Answer: " + response)
            audio = generate(
                text=response,
                voice="VOICE_NAME",
                model="eleven_multilingual_v2")
            print("Audio generated")
            play(audio)
    except Exception as err:
        print('Exception occurred. Please try again', str(err))