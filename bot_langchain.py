from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate

llm = ChatOllama(model="llama3.2:1b")
memory = ConversationBufferMemory(return_messages=True)

prompt = ChatPromptTemplate(
    messages=[
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

print("Conversation started. Type your message below:")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the conversation.")
        break
    
    response = conversation.run(input=user_input)
    print(f"Bot: {response}")