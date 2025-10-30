import torch
from torch.utils.data import DataLoader

def get_dataloaders(train_ds, val_ds, batch_size=16):
    """
    Returns PyTorch DataLoaders for train and validation sets.
    """
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader

def check_batch(dataloader, tokenizer):
    """
    Prints one batch to verify correctness.
    """
    batch = next(iter(dataloader))
    print("Keys:", batch.keys())
    print("Input IDs shape:", batch["input_ids"].shape)
    print("Attention Mask shape:", batch["attention_mask"].shape)
    print("Labels shape:", batch["labels"].shape)
    print("Decoded[0]:", tokenizer.decode(batch["input_ids"][0]))



import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score
from indicnlp.tokenize import indic_tokenize
from bert_score import score as bert_scorer
from sacrebleu.metrics import CHRF
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import re


def compute_bleu(references, predictions, lang='hi'):
    """
    Compute BLEU for Hindi (or other Indic languages) using proper Indic tokenization.
    """
    smooth = SmoothingFunction().method1
    tokenized_refs = [[indic_tokenize.trivial_tokenize(ref)] for ref in references]
    tokenized_preds = [indic_tokenize.trivial_tokenize(pred) for pred in predictions]
    
    #temp to test
    # if references and predictions:
    #     print("\nExample Hindi tokenization check:")
    #     print("Ref:", indic_tokenize.trivial_tokenize(references[0]))
    #     print("Pred:", indic_tokenize.trivial_tokenize(predictions[0]))
    
    return corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=smooth)


def compute_rouge(references, predictions, lang='hi'):
    """
    Compute ROUGE-L for Hindi using Indic tokenization and sentence-level aggregation.
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scores = []
    # if references and predictions:
    #     print("\nExample Hindi tokenization check:")
    #     print("Ref:", indic_tokenize.trivial_tokenize(references[0]))
    #     print("Pred:", indic_tokenize.trivial_tokenize(predictions[0]))

    for ref, pred in zip(references, predictions):
        ref_tok = " ".join(indic_tokenize.trivial_tokenize(ref))
        pred_tok = " ".join(indic_tokenize.trivial_tokenize(pred))
        score = scorer.score(ref_tok, pred_tok)
        scores.append(score['rougeL'].fmeasure)

    return sum(scores) / len(scores) if scores else 0.0

def compute_accuracy(true_labels, pred_labels):
    return accuracy_score(true_labels, pred_labels)

def compute_bertscore(references, predictions, lang='hi'):
    P, R, F1 = bert_scorer(predictions, references, lang=lang, model_type='xlm-roberta-large')
    return float(F1.mean())

def compute_perplexity(sentences, model_name="aashay96/indic-gpt"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    encodings = tokenizer("\n\n".join(sentences), return_tensors="pt")
    max_length = model.config.n_positions
    stride = 512
    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * stride
        lls.append(log_likelihood)
    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl.item()


def compute_chrf(references, predictions):
    """Compute chrF++ (best for morphologically rich Hindi)."""
    chrf = CHRF(word_order=2)
    return chrf.corpus_score(predictions, [references]).score / 100

def compute_similarity(references, predictions):
    """Compute cosine similarity between Hindi rationales."""
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ref_emb = model.encode(references, convert_to_tensor=True)
    pred_emb = model.encode(predictions, convert_to_tensor=True)
    return util.cos_sim(ref_emb, pred_emb).diagonal().mean().item()


# Composite Scoring Framework
# -----------------------------------------
def compute_reasoning_score(true_labels, pred_labels, true_rationale, pred_rationale):
    accuracy = compute_accuracy(true_labels, pred_labels)
    rationale_sim = compute_similarity(true_rationale, pred_rationale)
    return (accuracy + rationale_sim) / 2


def compute_domain_score(true_labels, pred_labels, true_advisor, pred_advisor, adequacy=0.8, actionability=0.8):
    """Adequacy/actionability can be human or rule-based scores."""
    accuracy = compute_accuracy(true_labels, pred_labels)
    # correctness = compute_bertscore(true_advisor, pred_advisor, lang='hi')
    advisory_similarity = compute_similarity(true_advisor, pred_advisor)
    return  (accuracy+advisory_similarity)/2 #(adequacy + correctness + actionability) / 3


def compute_language_score(true_advisor, pred_advisor):
    chrf_score = compute_chrf(true_advisor, pred_advisor)
    # rouge_score_val = compute_rouge(true_advisor, pred_advisor)
    bert_score_val = compute_bertscore(true_advisor, pred_advisor)
    return (chrf_score + bert_score_val) / 2


def compute_final_score(true_labels, pred_labels, true_rationale, pred_rationale, true_advisor, pred_advisor):
    if not true_labels or not pred_labels:
        return {"ReasoningScore": 0.0, "LanguageScore": 0.0, "FinalScore": 0.0}

    ReasoningScore = compute_reasoning_score(true_labels, pred_labels, true_rationale, pred_rationale)
    DomainScore = compute_domain_score(true_labels, pred_labels, true_advisor, pred_advisor)
    LanguageScore = compute_language_score(true_advisor, pred_advisor)
    FinalScore = 0.4 * ReasoningScore + 0.2 * LanguageScore + 0.4 * DomainScore

    return {
        "ReasoningScore": ReasoningScore,
        "DomainScore": DomainScore,
        "LanguageScore": LanguageScore,
        "FinalScore": FinalScore
    }