import numpy as np
from itertools import combinations
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')


import torch.nn.functional as F

from torch import Tensor
import torch
from transformers import AutoTokenizer, AutoModel

import numpy as np

from typing import Set

class Tokenizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("./thirdParty/multilingual-e5-large-instruct")
        self.model = AutoModel.from_pretrained("./thirdParty/multilingual-e5-large-instruct").to(self.device)
    
    def tokenize(self, input):
        return self.tokenizer(input, max_length=512, padding=True, truncation=True, return_tensors='pt')
    
    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def get_embeddings(self, input_words):
        word_embeddings = {}

        for word in input_words:
            inputs = self.tokenize(word).to(self.device)
            output = self.model(**inputs)

            last_hidden_states = output.last_hidden_state
            attention_mask = inputs['attention_mask']

            embeddings = self.average_pool(last_hidden_states, attention_mask)
            embeddings = embeddings.cpu().detach().numpy()

            # Remove the extra dimension
            embeddings = np.squeeze(embeddings, axis=0)

            word_embeddings[word] = embeddings

        return word_embeddings


class ConnectionsSolver:
    def __init__(self, words, beam_width=10):
        self.words = words.split(", ")
        self.beam_width = beam_width
        self.tokenizer = Tokenizer()
        self.word_embeddings = self.tokenizer.get_embeddings(self.words)

    def _get_word_embeddings(self):
        embeddings = {}
        with torch.no_grad():
            for word in tqdm(self.words, desc="Computing word embeddings"):
                inputs = self.tokenizer(word, return_tensors="pt")
                outputs = self.model(**inputs)
                embeddings[word] = outputs.last_hidden_state[0, 0, :].numpy()
        return embeddings

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def group_similarity_score(self, group):
        vectors = [self.word_embeddings[w] for w in group]
        
        # Clustering score
        kmeans = KMeans(n_clusters=1, random_state=42)
        kmeans.fit(vectors)
        cluster_score = -kmeans.inertia_

        # Minimum pairwise similarity
        pairwise_sims = [self.cosine_similarity(vectors[i], vectors[j]) 
                         for i, j in combinations(range(4), 2)]
        min_sim = min(pairwise_sims)

        # Variance-based score
        var_score = np.mean(pairwise_sims) / (1 + np.var(pairwise_sims))

        # WordNet-based score
        #wordnet_score = self.wordnet_similarity(group)

        return 0.4 * cluster_score + 0.3 * min_sim + 0.3 * var_score #+ 0.1 * wordnet_score
        #return 0.5 * min_sim + 0.5 * var_score
        #return min_sim

    def wordnet_similarity(self, group):
        synsets = [wordnet.synsets(word) for word in group]
        scores = []
        for i, s1 in enumerate(synsets):
            for s2 in synsets[i+1:]:
                if s1 and s2:  # Check if both words have synsets
                    max_sim = max((ss1.path_similarity(ss2) or 0)
                                   for ss1 in s1 for ss2 in s2)
                    scores.append(max_sim)
        return np.mean(scores) if scores else 0

    def calculate_penalty(self, group, other_words):
        group_vec = np.mean([self.word_embeddings[w] for w in group], axis=0)
        other_vecs = [self.word_embeddings[w] for w in other_words]
        similarities = [self.cosine_similarity(group_vec, ov) for ov in other_vecs]
        return np.mean(similarities) if similarities else 0

    def solve(self, exclude):
        beam = [([],set(self.words),0)]
        
        for step in tqdm(range(4), desc="Solving"):
            new_beam = []
            for solution, remaining, score in tqdm(beam, desc=f"Step {step+1}", leave=False):
                candidates = list(combinations(remaining, 4))
                exclude_set = set(tuple(sorted(group)) for group in exclude)
                candidates = [group for group in candidates if tuple(sorted(group)) not in exclude_set]
                for candidate in candidates:
                    new_solution = solution + [candidate]
                    new_remaining = remaining - set(candidate)
                    group_score = self.group_similarity_score(candidate)
                    penalty = self.calculate_penalty(candidate, new_remaining)
                    new_score = score + group_score - penalty
                    new_beam.append((new_solution, new_remaining, new_score))
            
            new_beam.sort(key=lambda x: x[2], reverse=True)
            if len(new_beam) > 0:
                beam = new_beam[:self.beam_width]
        
        return beam[0][0]


# Example usage
# words = ["QUIET", "MAGIC", "COLD", "SIN", 
#          "BUG", "BREACH", "EASY", "ENOUGH", 
#          "SING", "WINDY", "DIVE", "MOTOR", 
#          "RELAX", "COUGH", "SPOUT", "CHILL"]
# words = "KIND, DRIFT, TENDER, NICE, IDEA, WING SORT, RING, TYPE, STICK, STYLE, SWEET, MESSAGE, SICK, COOL, POINT"

# # # tokenizer experiments
# # # tokenizer = Tokenizer()
# # # print(tokenizer.get_embeddings(words))

# solver = ConnectionsSolver(words)
# solution = solver.solve([])
# print(solution)