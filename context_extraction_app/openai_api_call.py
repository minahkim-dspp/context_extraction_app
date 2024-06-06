from openai import OpenAI

def openai_api (message : list, model = "gpt-3.5-turbo-0125", temperature = 0) -> OpenAI:
    '''
    openai_api calls the OPEN AI API.

    Parameter:
        message (dict) : a dictionary to send to the OpenAI. A common format is {"role": "user", "content": "question to ask"}
        model (str) : the model to use for OpenAi. Default as "gpt-3.5-turbo-0125"
        temperature(int) : the temperature for the OpenAI model. Default is 0

    Return:
        chat_completion (OpenAI): the object that has the OpenAI result
    '''

    client = OpenAI()

    chat_completion = client.chat.completions.create(
        messages= message,
        model = model,
        temperature = temperature
    )

    return chat_completion

    
    


    
    


