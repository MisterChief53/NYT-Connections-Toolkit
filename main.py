from chat import chat
from dotenv import load_dotenv
import random
import json
import re
import copy
from beam_search import ConnectionsSolver
from tqdm import tqdm
import typer
from enum import Enum

class Mode(str, Enum):
    SIMULATION = "simulation"
    ONE_TRY = "one-try"

class Model(str, Enum):
    NONE = "none"
    CLAUDE = "claude"
    GPT4 = "GPT-4"
    GPT4O = "GPT-4o"
    GOOGLE = "Google"
    LLAMA = "llama"
    LLAMA_405B = "llama 3.1 405b"
    QWEN = "qwen"
    MIXTRAL = "mixtral"

class Method(str, Enum):
    IO = "IO"
    COT = "CoT"
    COT_SC = "CoT-SC"
    HEURISTIC = "heuristic"
    RANDOM = "random"

class Hints(str, Enum):
    FULL = "full"
    NONE = "none"

def json_add_quotes(data):
    # This pattern matches brackets containing the list of words
    pattern = re.compile(r'(\[[^\[\]]*\])')

    # Function to add quotes around entries and handle separated words
    def add_quotes(match):
        inner = match.group(0)
        # First, replace improperly separated words with a proper single space
        inner = re.sub(r'"\s*"\s*', ' ', inner)
        # Add quotes around sequences of characters that are not enclosed in quotes
        corrected = re.sub(r'(?<!")\b([A-Za-z0-9_\-][A-Za-z0-9_\-\s]*[A-Za-z0-9_\-])\b(?!\")', r'"\1"', inner)
        return corrected

    # Apply the add_quotes function to each matched list
    corrected_data = pattern.sub(add_quotes, data)
    return corrected_data

def is_valid_json(data):
    try:
        json.loads(data)
        return True
    except:
        print("This is not valid json")
        return False 

def load_data(dataset_path: str):
    with open(dataset_path, "r") as file:
        data = json.load(file)


    answers = []
    for i in range(0, len(data)):
        groups = []
        for j in range(0, len(data[i]["answers"])):
            group = data[i]["answers"][j][0]
            input_string = data[i]["answers"][j][1][0]
            clean_string = input_string.lstrip('-').strip()
            words = [word.strip() for word in clean_string.split(',')]

            wordsSet = set()
            for word in words:
                wordsSet.add(word)

            groups.append(group)
            groups.append(wordsSet)
        answers.append(groups)

    #print(answers)

    return answers

def cot_message(chat, input):
    dirtyOutput = chat.message(input)

    json_start_index = dirtyOutput.find("{")
    return dirtyOutput[json_start_index:]

def extract_json(text):
    stack = []
    start_index = None
    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_index = i
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:
                    json_text = text[start_index:i + 1]
                    try:
                        return json.loads(json_text)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Found JSON-like text but failed to parse it: {e}")
    
    raise ValueError("No JSON object found in the text")

"""
mode: either "One-Try" or "Multiple-tries"
"""
def extract_array_json(text, mode):
    stack = []
    start_index = None
    # We need to handle both arrays and objects as possible JSON structures
    for i, char in enumerate(text):
        if char in '{[':  # Start of JSON object or array
            if not stack:
                start_index = i
            stack.append(char)
        elif char in '}]':  # End of JSON object or array
            if stack and ((stack[-1] == '{' and char == '}') or (stack[-1] == '[' and char == ']')):
                stack.pop()
                if not stack:  # When the stack is empty, we've captured a full JSON structure
                    json_text = text[start_index:i + 1]
                    try:
                        if not is_valid_json(json_text):
                            json_text = json_add_quotes(json_text)
                            if not is_valid_json(json_text):
                                if mode == "One-try":
                                    json_text = "[\n\t[\"none1\", \"none2\", \"none3\", \"none4\"],\n\t[\"none5\", \"none6\", \"none7\", \"none8\"],\n\t[\"none9\", \"none10\", \"none11\", \"none12\"],\n\t[\"none13\", \"none14\", \"none15\", \"none16\"]]"
                                elif mode == "Multiple-tries":
                                    json_text = "{\n\t\"group\": [\"none1\", \"none2\", \"none3\", \"none4\"]\n}"

                        data = json.loads(json_text)
                        # Verify that the data is in the expected format (list of lists)
                        if isinstance(data, list) and all(isinstance(sublist, list) for sublist in data):
                            return data
                        # Check if the data is a dictionary with a specific key containing a list
                        if isinstance(data, dict) and "group" in data and isinstance(data["group"], list):
                            return data
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Found JSON-like text but failed to parse it: {e}")
    
    # there was no json in the text, so we return placeholder json
    if mode == "One-try":
        json_text = "[\n\t[\"none1\", \"none2\", \"none3\", \"none4\"],\n\t[\"none5\", \"none6\", \"none7\", \"none8\"],\n\t[\"none9\", \"none10\", \"none11\", \"none12\"],\n\t[\"none13\", \"none14\", \"none15\", \"none16\"]]"
    elif mode == "Multiple-tries":
        json_text = "{\n\t\"group\": [\"none1\", \"none2\", \"none3\", \"none4\"]\n}"

    return json.loads(json_text)

def answers_equal(text1, text2, mode):
    input1 = json.loads(text1)
    input2 = json.loads(text2)

    output = False
    if mode == "One-try":
        # for each group in an answer, we need to find if
        # there is an equal group on the other one-> n^2
        # but first we convert all groups to sets
        groups1 = []
        groups2 = []
        for i in range(0, max(len(input1), len(input2))):
            if i < len(input1):
                groups1.append(set(input1[i]))
            if i < len(input2):
                groups2.append(set(input2[i]))

        # now we compare sets
        matches = 0
        for group1 in groups1:
            for group2 in groups2:
                if group1 == group2:
                    matches += 1
        
        output = True if matches == 4 else False
    elif mode == "Multiple-tries":
        output = True if set(input1['group']) == set(input2['group']) else False

    return output

def output_in_answers(output, answers, mode):
    for answer, (count, messages) in answers.items():
        if answers_equal(output, answer, mode):
            return (True, answer)
    
    return (False, "")

def cot_sc(input, chat, samples, mode):
    original_messages = copy.deepcopy(chat.get_messages())

    # "answer": [score, messages]
    answers = {}

    print(f"System: Trying {samples} samples...")
    with open("results.txt", 'a') as file:
        print(f"System: Trying {samples} samples...", file=file)

    for i in tqdm(range(0, samples), desc="Sampling..."):
        llmOutput = json.dumps(extract_array_json(chat.message(input), mode=mode))

        outputInAnswers, key = output_in_answers(llmOutput, answers, mode)
        if outputInAnswers:
            answers[key][0] += 1
        else:
            answers[llmOutput] = [1, copy.deepcopy(chat.get_messages())]
        
        chat.set_messages(original_messages)

    max = 0
    for answer, (count, messages) in answers.items():
        if count > max:
            finalAnswer = answer
            max = count

    chat.set_messages(answers[finalAnswer][1])

    print(f"final answer is:\n {finalAnswer} \n with a score of: {max}")

    return json.loads(finalAnswer)

def oneTry(model: str, method: str, samples: int, answers: list):
    print(f"\nStarting One Try simulation with {method} method")
    results = []

    for i in range(0, samples):
        matches = 0

        if method == "IO":
            # Load prompts from json file
            with open('prompts.json', 'r') as f:
                prompts_data = json.load(f)
            
            # Get the system prompt and messages from oneTry section
            prompt_data = next(
                prompt 
                for prompt in prompts_data["oneTry"] 
                if prompt["PromptingMethod"] == "IO"
            )
            system_prompt = prompt_data["systemPrompt"]
            messages = prompt_data["messages"]
            
            currentChat = chat(model=model, systemPrompt=system_prompt, messages=messages)
        elif method == "CoT" or method == "CoT-SC":
            # Load prompts from json file
            with open('prompts.json', 'r') as f:
                prompts_data = json.load(f)
            
            # Get the system prompt and messages from oneTry section for CoT
            prompt_data = next(
                prompt 
                for prompt in prompts_data["oneTry"] 
                if prompt["PromptingMethod"] == "CoT"
            )
            system_prompt = prompt_data["systemPrompt"]
            messages = prompt_data["messages"]
            
            currentChat = chat(model=model, systemPrompt=system_prompt, messages=messages)
            
        # initially extract words from the dataset
        words = []
        wordsSets = []
        # we iterate through the 4 groups (ignoring the first element)
        for j in range(1, len(answers[i])):
            # if we are in an odd index, we are in a group
            if j % 2 != 0:
                # add words on group to the words array
                group = answers[i][j]
                wordsSets.append(answers[i][j])
                for k in range(0, len(group)):
                    words.append(list(group)[k])
        
        random.shuffle(words)

        # ask about the current group
        llmInput = ', '.join(words)

        if method == "IO" or method == "CoT":
            llmOutput = currentChat.message(llmInput)
            if method == "IO":
                if not is_valid_json(llmOutput):
                    llmOutput = json_add_quotes(llmOutput)
                    if not is_valid_json(llmOutput):
                        llmOutput = "[\n\t[\"none1\", \"none2\", \"none3\", \"none4\"],\n\t[\"none5\", \"none6\", \"none7\", \"none8\"],\n\t[\"none9\", \"none10\", \"none11\", \"none12\"],\n\t[\"none13\", \"none14\", \"none15\", \"none16\"]]"
                llmOutput = json.loads(llmOutput)
            elif method == "CoT":
                llmOutput = extract_array_json(llmOutput, "One-try")
            
            print(f"final (in memory) llm output: {llmOutput}")
        elif method == "CoT-SC":
            llmOutput = cot_sc(chat=currentChat, input=llmInput, samples=10, mode="One-try")
        elif method == "heuristic":
            llmOutput = []
            solver = ConnectionsSolver(llmInput, 10)
            groups = solver.solve([])
            print(groups)
            for group in groups:
                llmOutput.append([group[0], group[1], group[2], group[3]])
            del solver

        for j in range(0, len(llmOutput)):
            for k in range(0, len(llmOutput)):
                count = 0
                for llmWord in llmOutput[j]:
                    if llmWord not in wordsSets[k]:
                        break
                    else:
                        count += 1
                
                if count == 4:
                    matches += 1
        
        results.append((matches/4) * 100)
    
    # save the results in a new line in the text file
    print(results)
    print(sum(results) / len(results))
    with open("results.txt", 'a') as file:
        print(results, file=file)
        print(sum(results) / len(results), file=file)


def simulation(model: str, method: str, samples: int, hints: str, answers: list):
    print(f"\nStarting Multiple Tries simulation with {method} method")
    print(f"Hints enabled: {hints}")

    with open("logs.txt", 'a') as file:
        print(f"Testing {model} using {method} on {samples} samples. Hints: {hints}", file=file)

    results = []
    # for each group of answers
    for i in range(0, samples):
        matches = 0
        mistakesRemaining = 3

        if method == "IO":
            with open('prompts.json', 'r') as f:
                prompts_data = json.load(f)

            prompt_data = next(
                prompt 
                for prompt in prompts_data["multipleTries"] 
                if prompt["PromptingMethod"] == "IO"
            )
            system_prompt = prompt_data["systemPrompt"]
            messages = prompt_data["messages"]
            
            currentChat = chat(model=model, systemPrompt=system_prompt, messages=messages)
        elif method == "CoT" or method == "CoT-SC":
            with open('prompts.json', 'r') as f:
                prompts_data = json.load(f)

            prompt_data = next(
                prompt 
                for prompt in prompts_data["multipleTries"] 
                if prompt["PromptingMethod"] == "CoT"
            )
            system_prompt = prompt_data["systemPrompt"]
            messages = prompt_data["messages"]
            
            currentChat = chat(model=model, systemPrompt=system_prompt, messages=messages)

        # initially extract words from json parsing
        words = []
        wordsSets = []
        # we iterate through the 4 groups (ignoring the first element)
        for j in range(1, len(answers[i])):
            # if we are in an odd index, we are in a group
            if j % 2 != 0:
                # add words on group to the words array
                group = answers[i][j]
                wordsSets.append(answers[i][j])
                for k in range(0, len(group)):
                    words.append(list(group)[k])
        
        random.shuffle(words)

        oneAway = False

        groupsToExclude = []
        
        while mistakesRemaining >= 0:
            # ask about the current group
            llmInput = ', '.join(words)

            if method != "heuristic":
                llmInput = llmInput + f'\nMistakes Remaining: {mistakesRemaining}'

            if oneAway:
                llmInput = llmInput + "\nOne Away"
                oneAway = False

            print(f"Input: {llmInput}")
            with open("logs.txt", 'a') as file:
                print(f"Input: {llmInput}", file=file)

            if method == "IO" or method == "CoT":
                llmOutput = currentChat.message(llmInput)
                with open("logs.txt", 'a') as file:
                    print(f"LLM Output: {llmOutput}", file=file)

                if method == "IO":
                    if not is_valid_json(llmOutput):
                        llmOutput = json_add_quotes(llmOutput)
                        if not is_valid_json(llmOutput):
                            llmOutput = "{\n\t\"group\": [\"none1\", \"none2\", \"none3\", \"none4\"]\n}"
                        print(f"Corrected output: {llmOutput}")
                    llmOutput = json.loads(llmOutput)
                elif method == "CoT":
                    llmOutput = extract_array_json(llmOutput, "Multiple-tries")
                print(f"final (in memory) llm output: {llmOutput}")
            elif method == "CoT-SC":
                llmOutput = cot_sc(chat=currentChat, input=llmInput, samples=10, mode="Multiple-tries")
                with open("logs.txt", 'a') as file:
                    print(f"LLM Output: {llmOutput}", file=file)
            elif method == "heuristic":
                llmOutput = {"group": []}
                solver = ConnectionsSolver(llmInput, 10)
                rawGroup = solver.solve(groupsToExclude)[0]
                print(f"rawGroup: {rawGroup}")
                for word in rawGroup:
                    llmOutput["group"].append(word)
                del solver
            elif method == "random":
                llmOutput = {"group": []}
                print(f"llmOutput: {llmOutput}")
                # if generated a group that is in "groups to exclude",
                # generate another one
                while llmOutput["group"] == [] or any(set(llmOutput["group"]) == set(group) for group in groupsToExclude):
                    llmOutput["group"] = []
                    tempWords = copy.deepcopy(list(words))
                    for _ in range(4):
                        randomWord = random.choice(tempWords)
                        llmOutput["group"].append(randomWord)
                        tempWords.remove(randomWord)
                print(f"Random Output: {llmOutput}")


            # check if the llm answer matches a group. If it does, add a match
            # else add a mistake
            numGroups = len(wordsSets)
            for j in range(0, len(wordsSets)):
                count = 0
                for llmWord in llmOutput["group"]:
                    if llmWord in wordsSets[j]:
                        count += 1
                
                if count != 4:
                    if count == 3 and hints == "full":
                        oneAway = True
                    else:
                        oneAway = False
                    print("System: This group does not match, we proceed...")
                    continue

                matches += 1
                wordsSets.remove(wordsSets[j])
                break

            # if we still have the same number of groups, we made a mistake
            if numGroups == len(wordsSets):
                print("System: Made a mistake! Deducting from mistakes remaining...")
                if method == "heuristic" or method == "random":
                    group = llmOutput["group"]
                    groupSet = set()
                    for word in group:
                        groupSet.add(word)
                    groupsToExclude.append(groupSet)
                    print(f"System: groups to exclude: {groupsToExclude}")

                mistakesRemaining -= 1

            # we have won
            if len(wordsSets) == 0:
                print("System: WordsSets Empty, we have won!")
                break

            # now recapture the words array for the next iteration
            words = []
            for wordSet in wordsSets:
                for k in range(0, len(wordSet)):
                    words.append(list(wordSet)[k])
            
            random.shuffle(words)
        
        #print((matches/4) * 100)
        results.append((matches/4) * 100)

    print(results)
    print(sum(results) / len(results))
    with open("results.txt", 'a') as file:
        print(results, file=file)
        print(sum(results) / len(results), file=file)


def main(
    mode: Mode = typer.Option(..., help="Mode to run: 'simulation' or 'one-try'"),
    model: Model = typer.Option(Model.NONE, help="Model to use for LLM inference"),
    method: Method = typer.Option(Method.HEURISTIC, help="Method to use for solving"),
    samples: int = typer.Option(25, help="Number of samples to run", min=1),
    hints: Hints = typer.Option(Hints.FULL, help="Hint mode for simulation"),
    dataset: str = typer.Option("full-hints.json", help="Path to dataset JSON file")
):
    """
    Run Connections game solver in different modes.
    
    Examples:
        python main.py --mode simulation --model claude --method cot --samples 10 --hints full
        python main.py --mode one-try --model gpt4 --method io --samples 25
    """
    load_dotenv()
    answers = load_data(dataset)

    # Convert enum values to strings for compatibility with existing functions
    model_str = model.value
    method_str = method.value
    hints_str = hints.value

    if mode == Mode.SIMULATION:
        simulation(model=model_str, method=method_str, samples=samples, hints=hints_str, answers=answers)
    elif mode == Mode.ONE_TRY:
        oneTry(model=model_str, method=method_str, samples=samples, answers=answers)

if __name__ == "__main__":
    typer.run(main)