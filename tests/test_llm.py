import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm_provider import ChatBackend, OpenAIBackend, TogetherAIBackend
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
together_key = os.getenv("TOGETHER_API_KEY")

# model: gpt-4o, gpt-5-mini, openai/gpt-oss-120b, openai/gpt-oss-20b
agent1 = OpenAIBackend(model="gpt-5-mini", key=openai_key, url="https://api.openai.com/v1")
agent2 = OpenAIBackend(model="openai/gpt-oss-20b", key=together_key, url="https://api.together.xyz/v1")

prompt1 = """Generate the triplets for the following text: 
                Robert Koch was a German physician and microbiologist.
            
            Save the triplets in the format of (subject, predicate, object).
            Finanlize your answer in json format like:
            [
                {"subject": "", 
                "relation": "", 
                "object": ""}
            ]
        """

result = agent1.chat(messages=[{"role": "user", "content": prompt1}])
result2 = agent2.chat(messages=[{"role": "user", "content": prompt1}])

print(result)
print(result2)