#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: mrc_ner_inference.py

import os
import torch
import argparse
from torch.utils.data import DataLoader
from utils.random_seed import set_random_seed
set_random_seed(0)
from train.mrc_ner_trainer import BertLabeling
from tokenizers import BertWordPieceTokenizer
from dataset.mrc_ner_dataset import MRCNERDataset
from metrics.functional.query_span_f1 import extract_flat_spans, extract_nested_spans

def get_dataloader(config, data_prefix="test"):
    data_path = os.path.join(config.data_dir, f"mrc-ner.{data_prefix}")
    vocab_path = os.path.join(config.bert_dir, "vocab.txt")
    data_tokenizer = BertWordPieceTokenizer(vocab_path)

    dataset = MRCNERDataset(json_path=data_path,
                            tokenizer=data_tokenizer,
                            max_length=config.max_length,
                            is_chinese=config.is_chinese,
                            pad_to_maxlen=False)

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    return dataloader, data_tokenizer

def get_query_index_to_label_cate(dataset_sign):
    # NOTICE: need change if you use other datasets.
    # please notice it should in line with the mrc-ner.test/train/dev json file
    if dataset_sign == "conll03":
        return {1: "ORG", 2: "PER", 3: "LOC", 4: "MISC"}
    elif dataset_sign == "ace04":
        return {1: "GPE", 2: "ORG", 3: "PER", 4: "FAC", 5: "VEH", 6: "LOC", 7: "WEA"}
    elif dataset_sign == "az33":
        return {0: 'ActiveIngredients', 1: 'AgeRangeDescription', 2: 'BatteryCellComposition', 3: 'Brand', 4: 'CaffeineContent', 5: 'CapacityUnit', 6: 'CoffeeRoastType', 7: 'Color', 8: 'DietType', 9: 'DosageForm', 10: 'EnergyUnit', 11: 'FinishType', 12: 'Flavor', 13: 'FormulationType', 14: 'HairType', 15: 'Ingredients', 16: 'ItemForm', 17: 'ItemShape', 18: 'LiquidContentsDescription', 19: 'Material', 20: 'MaterialFeature', 21: 'MaterialTypeFree', 22: 'PackageSizeName', 23: 'Pattern', 24: 'PatternType', 25: 'ProductBenefit', 26: 'Scent', 27: 'SkinTone', 28: 'SkinType', 29: 'SpecialIngredients', 30: 'TargetGender', 31: 'TeaVariety', 32: 'Variety'}
    elif dataset_sign == "ae48":
        return {0: 'ApplicablePlace', 1: 'AthleticShoeType', 2: 'BackSideMaterial', 3: 'BodyMaterial', 4: 'BrandName', 5: 'Capacity', 6: 'Category', 7: 'ClosureType', 8: 'Collar', 9: 'Color', 10: 'DepartmentName', 11: 'DerivativeSeries', 12: 'FabricType', 13: 'Feature', 14: 'FingerboardMaterial', 15: 'Fit', 16: 'Function', 17: 'Gender', 18: 'HoseHeight', 19: 'InsoleMaterial', 20: 'IsCustomized', 21: 'ItemType', 22: 'Length', 23: 'LensesOpticalAttribute', 24: 'LevelOfPractice', 25: 'LiningMaterial', 26: 'Material', 27: 'Model', 28: 'ModelNumber', 29: 'Name', 30: 'OuterwearType', 31: 'OutsoleMaterial', 32: 'PatternType', 33: 'ProductType', 34: 'Season', 35: 'Size', 36: 'SleeveLengthCm', 37: 'SportType', 38: 'SportsType', 39: 'StrapType', 40: 'Style', 41: 'Technology', 42: 'Type', 43: 'TypeOfSports', 44: 'UpperHeight', 45: 'UpperMaterial', 46: 'Voltage', 47: 'Weight'}

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="inference the model output.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--bert_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--is_chinese", action="store_true")
    parser.add_argument("--model_ckpt", type=str, default="")
    parser.add_argument("--hparams_file", type=str, default="")
    parser.add_argument("--flat_ner", action="store_true",)
    parser.add_argument("--dataset_sign", type=str, choices=["ontonotes4", "msra", "conll03", "ace04", "ace05","az33", "ae48"], default="conll03")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    trained_mrc_ner_model = BertLabeling.load_from_checkpoint(
        checkpoint_path=args.model_ckpt,
        hparams_file=args.hparams_file,
        map_location=None,
        batch_size=1,
        max_length=args.max_length,
        workers=0)

    data_loader, data_tokenizer = get_dataloader(args,)
    # load token
    vocab_path = os.path.join(args.bert_dir, "vocab.txt")
    with open(vocab_path, "r") as f:
        subtokens = [token.strip() for token in f.readlines()]
    idx2tokens = {}
    for token_idx, token in enumerate(subtokens):
        idx2tokens[token_idx] = token

    query2label_dict = get_query_index_to_label_cate(args.dataset_sign)

    for batch in data_loader:
        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch
        attention_mask = (tokens != 0).long()

        start_logits, end_logits, span_logits = trained_mrc_ner_model.model(tokens, attention_mask=attention_mask, token_type_ids=token_type_ids)
        start_preds, end_preds, span_preds = start_logits > 0, end_logits > 0, span_logits > 0

        subtokens_idx_lst = tokens.numpy().tolist()[0]
        subtokens_lst = [idx2tokens[item] for item in subtokens_idx_lst]
        label_cate = query2label_dict[label_idx.item()]
        readable_input_str = data_tokenizer.decode(subtokens_idx_lst, skip_special_tokens=True)

        if args.flat_ner:
            entities_info = extract_flat_spans(torch.squeeze(start_preds), torch.squeeze(end_preds),
                                               torch.squeeze(span_preds), torch.squeeze(attention_mask), pseudo_tag=label_cate)
            entity_lst = []

            if len(entities_info) != 0:
                for entity_info in entities_info:
                    start, end = entity_info[0], entity_info[1]
                    entity_string = " ".join(subtokens_lst[start: end])
                    entity_string = entity_string.replace(" ##", "")
                    entity_lst.append((start, end, entity_string, entity_info[2]))

        else:
            match_preds = span_logits > 0
            entities_info = extract_nested_spans(start_preds, end_preds, match_preds, start_label_mask, end_label_mask, pseudo_tag=label_cate)

            entity_lst = []

            if len(entities_info) != 0:
                for entity_info in entities_info:
                    start, end = entity_info[0], entity_info[1]
                    entity_string = " ".join(subtokens_lst[start: end+1 ])
                    entity_string = entity_string.replace(" ##", "")
                    entity_lst.append((start, end+1, entity_string, entity_info[2]))

        print("*="*10)
        print(f"Given input: {readable_input_str}")
        print(f"Model predict: {entity_lst}")
        # entity_lst is a list of (subtoken_start_pos, subtoken_end_pos, substring, entity_type)

if __name__ == "__main__":
    main()