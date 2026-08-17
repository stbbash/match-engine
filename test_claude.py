import anthropic
from dotenv import load_dotenv
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# message = client.messages.create(
#     model="claude-opus-4-6",
#     max_tokens=1024,
#     system="You are a helpful coding assistant specializing in Python.",
#     messages=[
#         {"role": "user", "content": "How do I sort a list of dictionaries by key?"}
#     ],
# )
# print(message.content)


# import base64
# import httpx

# image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
# image_media_type = "image/jpeg"
# image_data = base64.standard_b64encode(httpx.get(image_url).content).decode("utf-8")

# client = anthropic.Anthropic()

# response = client.messages.count_tokens(
#     model="claude-opus-4-6",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "source": {
#                         "type": "base64",
#                         "media_type": image_media_type,
#                         "data": image_data,
#                     },
#                 },
#                 {"type": "text", "text": "Describe this image"},
#             ],
#         }
#     ],
# )
# print(response.json())

def max_rot(n):
    numbers = [n]
    s = str(n)

    for i in range(len(s) - 1):
        # fixed part
        left = s[:i]

        # part to rotate
        right = s[i:]

        # rotate left
        right = right[1:] + right[0]

        # combine back
        s = left + right

        numbers.append(int(s))

    return max(numbers)


print(max_rot(56789))      # 68957
print(max_rot(38458215))   # 85821534