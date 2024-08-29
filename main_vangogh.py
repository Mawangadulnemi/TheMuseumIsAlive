import funs_vangogh
import time
from transformers import AutoTokenizer, AutoModel

while True:

  # If bigger than 0.5 in similarity
  question=input("반 고흐에게 무슨 말을 하고싶으신가요? : ")  # question="요즘 너무 힘들어요. 저는 그냥 쉬고싶어요."  # 사용자 input

  # Run AI ChatBot
  return_ai=funs_vangogh.run(question)
  print(f"return from Shin Hae Chul\n\n{return_ai}")
  # print(f"Memory Check {funs_vangogh.memories}")