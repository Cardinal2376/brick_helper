import os
from volcenginesdkarkruntime import Ark
import base64
import requests
import mimetypes
from openai import OpenAI
import json
import time
from common import csv_to_matrix


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
# 初始化Ark客户端，从环境变量中读取您的API Key
client = Ark(
    # 此为默认路径，您可根据业务所在地域进行配置
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
    api_key=os.environ.get("ARK_API_KEY"),
)

def get_decription_csv(image_path, output_file_name):

    # 将图片转为Base64编码
    base64_image = encode_image(image_path)

    with open("template.txt", "r") as f:
        prompt1 = f.read()
    response = client.chat.completions.create(
        # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
        model="ep-20250412201458-dqvzm",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":  f"data:image/jpg;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": f"{prompt1}"},
                ],
            }
        ],

    )
    print("prompt send.")
    content = response.choices[0].message.content
    # print(content)
    with open(f"{output_file_name}", "w") as f:
        f.write(content)
    print(f"content saved at {output_file_name}")


def sensenova():


    client = OpenAI(
        api_key="sk-ZEMJbDQ6IxzTex7UMiaaOF21Es85shv6",
        base_url="https://api.sensenova.cn/v1/llm/chat-completions"
    )
    image_path = "data/debug_2025-04-12_20-06-33.jpg"
    base64_image = encode_image(image_path)

    with open("template.txt", "r") as f:
        prompt1 = f.read()
    response = client.chat.completions.create(
        model="SenseChat-Vision",
        messages=[
            # {"role": "system",
            #  "content": "请你扮演一名优秀的故事创作者，并按照以下步骤完成故事创作：1.根据用户的要求设定【故事主题】。2.按照以下格式输出：【故事主题】xxx、【故事名称】xxx、【故事内容】xxx。3.输出的内容不要超过500个字。"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": f"{prompt1}"},
                ],
            }
        ],
        top_p=0.7,
        temperature=1.0
    )
    content = response.choices[0].message.content
    print(content)
    with open(f"data/sensevision_1.csv", "w") as f:
        f.write(content)
    print(f"content saved.")



def upload_image(file_path):

    api_token = os.getenv('SENSECHAT_API_KEY')

    url = "https://file.sensenova.cn/v1/files"
    headers = {"Authorization": f"Bearer {api_token}"}


    description = "image"
    scheme = "MULTIMODAL_1"

    # 验证文件存在性
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件 '{file_path}' 不存在")

    # 自动检测MIME类型
    filename = os.path.basename(file_path)
    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type:
        mime_type = 'application/octet-stream'  # 默认类型

    try:
        with open(file_path, 'rb') as f:
            files = [
                ('description', (None, description)),
                ('scheme', (None, scheme)),
                ('file', (filename, f, mime_type))
            ]
            response = requests.post(url, headers=headers, files=files)
            response.raise_for_status()  # 自动抛出HTTP错误
            result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        raise
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        raise
    print(result)
    return result


def get_decription_csv_sensenova(image_path, output_file_name):
    # image_path = "data/debug_2025-04-12_20-58-11.jpg"
    # result = upload_image(file_path)
    # status = result['status']
    # image_id = result['id']
    # if status == 'VALID':
    # 从环境变量中获取 API_TOKEN
    base64_image = encode_image(image_path)
    # model_id = "SenseNova-V6-Pro"
    model_id = "SenseNova-V6-Reasoner"
    # model_id = "SenseChat-5-1202"
    API_TOKEN = os.getenv('SENSECHAT_API_KEY')

    with open("template.txt", "r") as f:
        prompt1 = f.read()

    # 定义请求的 URL
    url = "https://api.sensenova.cn/v1/llm/chat-completions"

    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }

    # 定义请求体
    data = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_base64",
                        "image_base64": f"{base64_image}"
                    },
                    {
                        "type": "text",
                        "text": f"{prompt1}"
                    }
                ]
            }
        ],
        "max_new_tokens": 4096,
        "repetition_penalty": 1.0,
        "stream": True,
        "temperature": 0.8,
        "top_p": 0.7,
        "user": "string"
    }

    try:
        # 发送 POST 请求
        last_time = time.time()
        print("requests sent")
        response = requests.post(url, headers=headers, json=data, stream=False)

        # 检查响应状态码
        response.raise_for_status()


        # 解析响应 JSON 数据
        # response_data = response.json()

        # 打印响应数据
        # print(response_data)

        result_text = ""
        reasoning_text = ""
        first = True
        for line in response.iter_lines():
            if first:
                print("ttft ", time.time() - last_time)
                first = False
            if line.startswith(b"data:"):
                # 去除 "data:" 前缀
                json_data = line[5:].decode("utf-8")
                if json_data == "[DONE]":
                    break
                try:
                    data = json.loads(json_data)
                    # print(data)
                    delta = data["data"]["choices"][0]["delta"]
                    if "reasoning_content" in data["data"]["choices"][0]:
                        reasoning_content = data["data"]["choices"][0]["reasoning_content"]
                    else:
                        reasoning_content = ""
                    reasoning_text += reasoning_content
                    result_text += delta
                    print(reasoning_content + delta, end="", flush=True)
                    # print(delta, end=" ", flush=True)
                except json.JSONDecodeError:
                    print("Failed to decode JSON data:", json_data)
            response.close()
        print("all time", time.time() - last_time)
        cleaned_response = result_text.strip().replace("``` csv", "").replace("```", "").strip().replace("csv", "")
        print(cleaned_response)
        with open(f"{output_file_name}", "w") as f:
            f.write(cleaned_response)
        print(f"content saved.")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")


def get_description_sensenova_v2(image_path, output_file_name):
    import sensenova
    import sys
    # 创建会话
    stream = True  # 流式输出或非流式输出
    model_id = "SenseNova-V6-Reasoner"  # 填写真实的模型ID



    sensenova.access_key_id = os.getenv("SENSENOVA_ACCESS_KEY_ID")
    sensenova.secret_access_key = os.getenv("SENSENOVA_SECRET_ACCESS_KEY")
    # sensenova
    resp = sensenova.Model.list()
    # 获取http headers
    print(resp.headers(), resp)
    resp = sensenova.Model.retrieve(id=model_id)
    print(resp)

    resp = sensenova.ChatCompletion.create(
        messages=[{"role": "user",
                   "content": "Say this is a test!"}],
        model=model_id,
        stream=stream,
        max_new_tokens=4096,
        n=1,
        repetition_penalty=1.05,
        temperature=0.8,
        top_p=0.7,
        know_ids=[],
        user="string",
    )


    if not stream:
        resp = [resp]
    for part in resp:
        # print(part)
        if stream:
            delta = part["data"]["choices"][0]["delta"]

            if "reasoning_content" in part["data"]["choices"][0]:
                reasoning_content = part["data"]["choices"][0]["reasoning_content"]
            else:
                reasoning_content = None
            if reasoning_content:
                sys.stdout.write(reasoning_content)
            if delta:
                sys.stdout.write(delta)

        else:
            sys.stdout.write(part["data"]["message"])
        sys.stdout.flush()


if __name__ == '__main__':
    # get_decription_csv("")
    # get_description_sensenova_v2("data/debug_split_0_0.jpg", "data/sensenova1.csv")
    get_decription_csv_sensenova("data/debug_split_0_0.jpg", "data/sensenova1.csv")
    # matrix, icon_name_matrix = csv_to_matrix("data/sensenova1.csv")
    # print(matrix)
    # print(icon_name_matrix)
    # for row in icon_name_matrix:
    #     for col in row:
    #         print(f"{col[:2]:<12}", end='')
    #     print()


