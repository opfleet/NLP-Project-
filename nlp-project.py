# imports need necessary installs found above

from naiveModel import NBLangIDModel
from BERTModel import BERTGenreClassification, train_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

from util import get_dataloader, BERT_preprocess

# don't need cuda until using virtual machine
from torch import manual_seed, cuda
import torch
from transformers import AutoTokenizer
from transformers import  AutoModelForSequenceClassification, Trainer, TrainerCallback, TrainingArguments
from datasets import Dataset
import numpy as np

from torch.utils.data import DataLoader

device = torch.device('cuda' if cuda.is_available() else 'cpu')

naiveBayes = NBLangIDModel()

# load data, train test split
descriptions = pd.read_csv("cleanedData.csv")

print("Shape before dropping NaN values:", descriptions.shape)
descriptions = descriptions.dropna()
print("Shape after dropping NaN values:", descriptions.shape)
train, test = train_test_split(descriptions, test_size=0.2)

train = train.drop("Unnamed: 0", axis= 1)
test = test.drop("Unnamed: 0", axis= 1)

train_X = train['description']
train_y1 = train['genre1']

# for top 3
#train_y2 = train['genre2']
#train_y3 = train['genre3']

# fit the NB model 
naiveBayes.fit(train_X.tolist(), train_y1.tolist())

# taking away leading whitespace
#train['genre1'] = train['genre1'].str.strip()
#train['genre2'] = train['genre2'].str.strip()
#train['genre3'] = train['genre3'].str.strip()

label_vocab = naiveBayes.labels

label_as_id = {l:k  for k, l in enumerate(label_vocab)}
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
bert_train, bert_val = train_test_split(train, test_size= 0.2)

raw_bert_train = Dataset.from_pandas(bert_train)
raw_bert_val = Dataset.from_pandas(bert_val)

ds = {'train': raw_bert_train, 'validation': raw_bert_val}

label_vocab = naiveBayes.labels
id2label = {k:l  for k, l in enumerate(label_vocab)}
label2id = {l:k  for k, l in enumerate(label_vocab)}

for split in ds:
    ds[split] = ds[split].map(lambda x: BERT_preprocess(x, id2label, tokenizer), remove_columns= ['description', 'genre1', 'genre2', 'genre3'])
    #ds[split] = DataLoader(ds[split], batch_size=batch_size)

BERT_preprocess({
    'description': "Blood sings to blood, Froi . . . Those born last will make the first . . . For Charyn will be barren no more. \n Three years after the curse on Lumatere was lifted, Froi has found his home... Or so he believes...Fiercely loyal to the Queen and Finnikin, Froi has been trained roughly and lovingly by the Guard sworn to protect the royal family, and has learned to control his quick temper. But when he is sent on a secretive mission to the kingdom of Charyn, nothing could have prepared him for what he finds. Here he encounters a damaged people who are not who they seem, and must unravel both the dark bonds of kinship and the mysteries of a half-mad Princess.And in this barren and mysterious place, he will discover that there is a song sleeping in his blood, and though Froi would rather not, the time has come to listen.Gripping and intense, complex and richly imagined, Froi of the Exiles is a dazzling sequel to Finnikin of the Rock, from the internationally best-selling and multi-award-winning author of Looking for Alibrandi, Saving Francesca, On the Jellicoe Road and The Piper's Son.",
    'genre1': 'Fantasy',
    'genre2': 'Young Adult',
    'genre3': 'Romance'
}, id2label, tokenizer)

#FUNCTIONS
def get_preds_from_logits(logits):
    ret = torch.zeros(logits.shape)
    logits = torch.from_numpy(logits)
    
    # for top 3
    #_, idx = logits.topk(3, dim=0, largest= True)

    # for top 1
    probs = torch.nn.functional.softmax(logits, dim= 0)
    _, idx = probs.topk(1, dim=0, largest= True)
    ret[idx] = 1.0
    return ret

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    final_metrics = {}

    predictions = get_preds_from_logits(logits)

    final_metrics['f1_micro'] = f1_score(labels, predictions, average= 'micro')
    final_metrics['f1_macro'] = f1_score(labels, predictions, average= 'macro')

    #Classification report
    print('Classification report:')
    print(classification_report(labels, predictions, zero_division = 0))

    return final_metrics

#BERT PARAMETERS
#LEARNING_RATE = 1e-4
LEARNING_RATE = 1e-4
MAX_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 25

#CLASSES
class MultiTaskClassificationTrainer(Trainer):
    def __init__(self, group_weights = None, **kwargs):
        super().__init__(**kwargs)
        self.group_weights = group_weights

    def compute_loss(self, model, inputs, return_outputs = False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs[0]
        
        #preds = get_preds_from_logits(logits.cpu().detach().numpy())
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

        #loss = torch.nn.functional.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
class PrinterCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        print(f'Epoch {state.epoch}: ')

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', id2label=id2label, label2id=label2id).to(device)

#MORE BERT PARAMETERS
training_args = TrainingArguments(
    output_dir= './distil-fine-tuned',
    learning_rate= LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    metric_for_best_model="f1_macro",
    load_best_model_at_end=True,
    weight_decay=0.01,
)

trainer = MultiTaskClassificationTrainer(
    model = model,
    args = training_args,
    train_dataset = ds['train'],
    eval_dataset = ds['validation'],
    compute_metrics = compute_metrics,
    callbacks = [PrinterCallback]
)

print("Training BERT model:")
trainer.train()

print("Evaluating BERT model:")
trainer.evaluate()

print("Testing BERT model:")
raw_bert_test = Dataset.from_pandas(test)
test_input_texts = raw_bert_test['description']

encoded = tokenizer(test_input_texts, truncation= True, padding= "max_length", max_length= 256, return_tensors= "pt").to(device)

# Call model to predict
logits = model(**encoded).logits.cpu().detach().numpy()

# Decode Logits
preds = get_preds_from_logits(logits.cpu().detach().numpy())
decoded_preds = [[id2label[i] for i, l in enumerate(row) if l == 1] for row in preds]

for text, pred in zip(test_input_texts, decoded_preds):
    print(text)
    print(decoded_preds)