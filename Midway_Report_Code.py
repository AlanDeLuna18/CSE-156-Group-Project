#Import needed packages 
import torch
from dataclasses import dataclass
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from datasets import load_dataset

'''
Overall idea, lmk if y'all like it: 
    -self_refine class: uses a model (maybe roberta?) to refine and learn from it's out output. Pretty much the class that does everything we need
        ->self.recent_refined_text = string that holds the most recently updated text
        ->self.model = roberta model (M) that will perform all the things we need (producing an output and making feedback)
        ->self.tokenizer = tokenizer that will help M do stuff 
        ->self.feedback = current feedback that will be used and sent back to self.model for refinement and start everything over again 
    -I'm making all of these notes based on this article just so that y'all know what I'm coming from lol: 
        -> https://cobusgreyling.medium.com/self-refine-is-an-iterative-refinement-loop-for-llms-23ffd598f8b8
'''

class self_refine: 
    def __init__(self):  
        self.refined_text = None 
        self.model = None 
        self.tokenizer = None
        self.feedback = None
    
    #Starts the whole process, takes in an input and feeds it to the model to start off the cycle (step 0 in diagram)
    def generator(file): 
        pass

    #Refines feedback and puts end result back into self.refined_text for future steps (steps 1 and 2 in diagram)
    def refinement():
        pass
        
    #Looks at self.refined_text and creates feedback based on whatever is in self.refined_text, puts end result in self.feedback (steps 3 and 4 in diagram)
    def feedback():
        pass


#Main loop, we could probably run expeiments here and see what happens, talk about it in the report and be like "oh this and this happened" lol




