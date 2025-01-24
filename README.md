Instructions:

For running the heuristic algorithm, `multilingual-e5-large-instruct` is needed. The easiest way to get it, is to head over to the hugging face hub https://huggingface.co/intfloat/multilingual-e5-large-instruct and clone the repository inside a folder in the root directory named "thirdParty". The final path should be:

`./thirdParty/multilingual-e5-large-instruct`

The configured prompts are held in prompts.json. Currently, it holds the initial prompts showcased in the paper.

Running the validator

To run the validator, you need to have poetry installed. Head over to the root folder, run `poetry shell`, and then `poetry install` to install the dependencies. 

Afterwards, you may run `python main.py --help` to see all the available arguments and their configurations, including *how to configure the dataset used*. It is important to note that the sample size is equal to the *number of samples* in the dataset. The runs will be logged in the `results.txt` and `logs.txt` files which will be created if needed.

The LLMs depend on certain API keys being configured correctly. You can find an example of what we expect in an `.env` file inside `.env.sample`, along with the cloud providers we've used. 