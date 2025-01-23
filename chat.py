import anthropic
from openai import OpenAI
import os
import google.generativeai as genai
import replicate
from huggingface_hub import InferenceClient
import copy


class chat:
    def __init__(self, model, systemPrompt, messages):
        print(f"\nLoading prompts for model: {model}")
        print(f"System Prompt: {systemPrompt if len(systemPrompt) <= 900 else systemPrompt[:900] + '...'}")
        print(f"Number of messages loaded: {len(messages)}")
        print("------------------------")
        
        self.model = model
        self.systemPrompt = systemPrompt
        self.messages = messages
        if(model == "claude"):
            self.client = anthropic.Anthropic()
        elif(model == "GPT-4" or model == "GPT-4o"):
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self.messages.insert(0, {"role": "system", "content": self.systemPrompt})
        elif(model == "llama 3.1 405b"):
            self.client = InferenceClient("meta-llama/Meta-Llama-3.1-405B-Instruct", token=os.environ.get("HUGGINGFACE_API_TOKEN"))
            print(os.environ.get("HUGGINGFACE_API_TOKEN"))
            self.messages.insert(0, {"role": "system", "content": self.systemPrompt})
        elif(model == "Google"):
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            self.client = genai.GenerativeModel("gemini-1.5-pro")
            newMessages = []
            for message in messages:
                newMessage = {}
                if message["role"] == "user":
                    newMessage = {"role": message["role"], "parts": [message["content"]]}
                elif message["role"] == "assistant":
                    newMessage = {"role": "model", "parts": [message["content"]]}
                newMessages.append(newMessage)
            self.messages = newMessages

            self.messages[0]["parts"][0] = self.systemPrompt + self.messages[0]["parts"][0]

    def message(self, userMessage):
        output = str()

        if(self.model == "claude"):
            self.messages.append({"role": "user", "content": userMessage})
            message = self.client.messages.create(
                model = "claude-3-5-sonnet-20240620",
                max_tokens = 1000,
                temperature = 0.5,
                system = self.systemPrompt,
                messages=self.messages
            )
            self.messages.append({"role": "assistant", "content": message.content[0].text})
            print(message.content[0].text)
            #output = json.loads(message.content[0].text)
            output = message.content[0].text
        elif(self.model == "GPT-4"):
            self.messages.append({"role": "user", "content": userMessage})
            message = self.client.chat.completions.create(
                model = "gpt-4",
                messages = self.messages,
                temperature=0.5
            )
            self.messages.append({"role": "assistant", "content": message.choices[0].message.content})
            print(message.choices[0].message.content)
            #output = json.loads(message.choices[0].message.content)
            output = message.choices[0].message.content
        elif(self.model == "GPT-4o"):
            self.messages.append({"role": "user", "content": userMessage})
            message = self.client.chat.completions.create(
                model = "gpt-4o",
                messages = self.messages,
                temperature=0.5
            )
            self.messages.append({"role": "assistant", "content": message.choices[0].message.content})
            print(message.choices[0].message.content)
            #output = json.loads(message.choices[0].message.content)
            output = message.choices[0].message.content
        elif(self.model == "Google"):
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]

            self.messages.append({"role": "user", "parts": [userMessage]})
            message = self.client.generate_content(
                self.messages,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    stop_sequences=['[x]'],
                    max_output_tokens=5000,
                    temperature=0.5
                ),
                safety_settings=safety_settings)
            print(message.text)
            self.messages.append({"role": "model", "parts": [message.text]})
            output = message.text
        elif(self.model == "llama"):
            self.messages.append({"role": "user", "content": userMessage})
            messageHistory = ""
            for i in range(0, len(self.messages)-1):
                messageRole = ""
                if self.messages[i]['role'] == "user":
                    messageRole = "user"
                elif self.messages[i]['role'] == "assistant":
                    messageRole = "assistant"

                escapedContent = self.messages[i]["content"].replace("{","{{").replace("}", "}}")
                messageHistory += "<|start_header_id|>" + messageRole + "<|end_header_id|>\n\n" + escapedContent + "<|eot_id|>"

            input = {
                "top_p": 0.9,
                "prompt": self.messages[len(self.messages)-1]["content"],
                "min_tokens": 0,
                "temperature": 0.6,
                "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + self.systemPrompt.replace("{", "{{").replace("}", "}}") + "<|eot_id|>" + messageHistory + "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "presence_penalty": 1.15
            }

            message = replicate.run(
                "meta/meta-llama-3-70b-instruct",
                input=input
            )

            print("".join(message))
            self.messages.append({"role": "assistant", "content": "".join(message)})
            output = "".join(message)
        elif(self.model == "llama 3.1 405b"):
            self.messages.append({"role": "user", "content": userMessage})
            messageHistory = ""
            for i in range(0, len(self.messages)-1):
                messageRole = ""
                if self.messages[i]['role'] == "user":
                    messageRole = "user"
                elif self.messages[i]['role'] == "assistant":
                    messageRole = "assistant"
                
                messageHistory += messageRole + ": " + self.messages[i]["content"] + "\n"
            
            input = {
                "prompt": self.systemPrompt + messageHistory + "user: " + self.messages[len(self.messages)-1]["content"] + "assistant: ",
                "max_tokens": 1024
            }

            message = replicate.run(
                "meta/meta-llama-3.1-405b-instruct",
                input=input
            )

            # print("".join(message))
            self.messages.append({"role": "assistant", "content": "".join(message)})
            output = "".join(message)
            print(output)
            # self.messages.append({"role": "user", "content": userMessage})

            # message = self.client.chat_completion(
            #     messages=self.messages,
            #     max_tokens=1024
            # )

            # self.messages.append({"role": "assistant", "content": message.choices[0].message.content})
            # print(message.choices[0].message.content)
            # output = message.choices[0].message.content
            # if(not self.is_valid_json(message.choices[0].message.content)):
            #     output = self.json_add_quotes(output)
        elif(self.model == "qwen"):
            self.messages.append({"role": "user", "content": userMessage})
            messageHistory = ""
            for i in range(0, len(self.messages)-1):
                messageRole = ""
                if self.messages[i]['role'] == "user":
                    messageRole = "user"
                elif self.messages[i]['role'] == "assistant":
                    messageRole = "system"

                escapedContent = self.messages[i]["content"]
                messageHistory += "\n" + messageRole + ":\n\n" + escapedContent + "\n"
            
            input = {
                "top_k": 50,
                "top_p": 0.9,
                "min_tokens": 0,
                "max_tokens": 500,
                "temperature": 0.6,
                "prompt": self.systemPrompt + messageHistory + "\n" + "user:\n\n" + self.messages[len(self.messages)-1]["content"] + "system:\n\n",
                "presence_penalty": 0,
                "frequency_penalty": 0
            }

            message = replicate.run(
                "lucataco/qwen2-72b-instruct:80759981bf24170e3acf8ec92982adfb161ca770ddecec1ba1a10eded801c9f3",
                input=input
            )
            print("Message:")
            print("".join(message))
            self.messages.append({"role": "assistant", "content": "".join(message)})
            output = "".join(message)
        elif(self.model == "mixtral"):
            self.messages.append({"role": "user", "content": userMessage})
            messageHistory = ""
            for i in range(0, len(self.messages)-1):
                messageRole = ""
                if self.messages[i]['role'] == "user":
                    messageRole = "user"
                elif self.messages[i]['role'] == "assistant":
                    messageRole = "assistant"

                escapedContent = self.messages[i]["content"].replace("{", "{{").replace("}", "}}")
                messageHistory += "\n" + messageRole + ":\n\n" + escapedContent + "\n"
            
            input = {
                "top_k": 50,
                "top_p": 0.9,
                "temperature": 0.6,
                "prompt": "user:\n\n" + self.messages[len(self.messages)-1]["content"] + "assistant:\n\n",
                "prompt_template:": self.systemPrompt.replace("{", "{{").replace("}", "}}") + messageHistory + "\n<s>[INST] {prompt} [/INST]",
            }

            message = replicate.run(
                "mistralai/mixtral-8x7b-instruct-v0.1",
                input=input
            )
            print("Message:")
            print("".join(message))
            self.messages.append({"role": "assistant", "content": "".join(message)})
            output = "".join(message)

        
        return output

    def get_messages(self):
        return self.messages

    def set_messages(self, messages):
        self.messages = copy.deepcopy(messages) # if not using deepcopy, we would point to messages instead
        if(self.model == "claude"):
            self.client = anthropic.Anthropic()
        elif(self.model == "OpenAI"):
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self.messages.insert(0, {"role": "system", "content": self.systemPrompt})
        elif(self.model == "Google"):
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            self.client = genai.GenerativeModel("gemini-1.5-pro")
            newMessages = []
            for message in messages:
                newMessage = {}
                if message["role"] == "user":
                    newMessage = {"role": message["role"], "parts": [message["content"]]}
                elif message["role"] == "assistant":
                    newMessage = {"role": "model", "parts": [message["content"]]}
                newMessages.append(newMessage)
            self.messages = newMessages

            self.messages[0]["parts"][0] = self.systemPrompt + self.messages[0]["parts"][0]