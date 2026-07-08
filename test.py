from openai import OpenAI

client = OpenAI(
    api_key="sk-gr-02f44cd3404411cd11cdd88da080685ac1c54986",
    base_url="https://endpoint.greatrouter.com" # 海外节点
    # base_url="https://endpoint.wendalog.com"                     # 国内备用节点
)

response = client.chat.completions.create(
    model="DeepSeek-V4-Flash",
    messages=[
        {
            "role": "user",
            "content": "hi"
        }
    ],
)

print(response.choices[0].message.content)