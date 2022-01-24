import os
import argparse

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from tokenizers import BertWordPieceTokenizer

from dataset.mrc_ner_dataset import MRCNERDataset
from train.mrc_ner_trainer import BertLabeling
from utils.get_parser import get_parser
from metrics.functional.query_span_f1 import extract_flat_spans, extract_nested_spans


def yd_add_parser(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--inference_file", type=str, required=True)
    parser.add_argument("--dataset_sign", type=str, choices=["ontonotes4", "msra", "conll03", "ace04", "ace05","az33"], default="conll03")
    return parser


def get_query_index_to_label_cate(dataset_sign):
    # NOTICE: need change if you use other datasets.
    # please notice it should in line with the mrc-ner.test/train/dev json file
    if dataset_sign == "conll03":
        return {1: "ORG", 2: "PER", 3: "LOC", 4: "MISC"}
    elif dataset_sign == "ace04":
        return {1: "GPE", 2: "ORG", 3: "PER", 4: "FAC", 5: "VEH", 6: "LOC", 7: "WEA"}
    elif dataset_sign == "az33":
        return {0: 'ActiveIngredients', 1: 'AgeRangeDescription', 2: 'BatteryCellComposition', 3: 'Brand', 4: 'CaffeineContent', 5: 'CapacityUnit', 6: 'CoffeeRoastType', 7: 'Color', 8: 'DietType', 9: 'DosageForm', 10: 'EnergyUnit', 11: 'FinishType', 12: 'Flavor', 13: 'FormulationType', 14: 'HairType', 15: 'Ingredients', 16: 'ItemForm', 17: 'ItemShape', 18: 'LiquidContentsDescription', 19: 'Material', 20: 'MaterialFeature', 21: 'MaterialTypeFree', 22: 'PackageSizeName', 23: 'Pattern', 24: 'PatternType', 25: 'ProductBenefit', 26: 'Scent', 27: 'SkinTone', 28: 'SkinType', 29: 'SpecialIngredients', 30: 'TargetGender', 31: 'TeaVariety', 32: 'Variety'}


def main():
    """main"""
    parser = get_parser()

    # add model specific args
    parser = BertLabeling.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    parser = yd_add_parser(parser)
    args = parser.parse_args()
    trained_mrc_ner_model = BertLabeling(args)
    args.is_chinese = False
    args.flat_ner = True

    if args.pretrained_checkpoint:
        print("yd: start to load ckpt !")
        trained_mrc_ner_model.load_state_dict(torch.load(args.pretrained_checkpoint,
                                         map_location=torch.device('cpu'))["state_dict"])
        print("yd: finish loading ckpt !")


    # obtain tokenizer
    vocab_path = os.path.join(args.bert_config_dir, "vocab.txt")
    with open(vocab_path, "r") as f:
        subtokens = [token.strip() for token in f.readlines()]
    idx2tokens = {}
    for token_idx, token in enumerate(subtokens):
        idx2tokens[token_idx] = token
    data_tokenizer = BertWordPieceTokenizer(vocab_path)

    # obtain dataset and dataloader
    dataset = MRCNERDataset(
        json_path=args.inference_file,
        tokenizer=data_tokenizer,
        max_length=args.max_length,
        is_chinese=args.is_chinese,
        pad_to_maxlen=False,
    )

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    # obtain query2label_dict (used by NER classes)
    query2label_dict = get_query_index_to_label_cate(args.dataset_sign)

    for batch in data_loader:
        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch
        attention_mask = (tokens != 0).long()

        start_logits, end_logits, span_logits = trained_mrc_ner_model.model(tokens, attention_mask=attention_mask,
                                                                            token_type_ids=token_type_ids)
        start_preds, end_preds, span_preds = start_logits > 0, end_logits > 0, span_logits > 0

        subtokens_idx_lst = tokens.numpy().tolist()[0]
        subtokens_lst = [idx2tokens[item] for item in subtokens_idx_lst]
        label_cate = query2label_dict[label_idx.item()]
        readable_input_str = data_tokenizer.decode(subtokens_idx_lst, skip_special_tokens=True)

        if args.flat_ner:
            entities_info = extract_flat_spans(torch.squeeze(start_preds), torch.squeeze(end_preds),
                                               torch.squeeze(span_preds), torch.squeeze(attention_mask),
                                               pseudo_tag=label_cate)
            entity_lst = []

            if len(entities_info) != 0:
                for entity_info in entities_info:
                    start, end = entity_info[0], entity_info[1]
                    entity_string = " ".join(subtokens_lst[start: end])
                    entity_string = entity_string.replace(" ##", "")
                    entity_lst.append((start, end, entity_string, entity_info[2]))

        else:
            match_preds = span_logits > 0
            entities_info = extract_nested_spans(start_preds, end_preds, match_preds, start_label_mask, end_label_mask,
                                                 pseudo_tag=label_cate)

            entity_lst = []

            if len(entities_info) != 0:
                for entity_info in entities_info:
                    start, end = entity_info[0], entity_info[1]
                    entity_string = " ".join(subtokens_lst[start: end + 1])
                    entity_string = entity_string.replace(" ##", "")
                    entity_lst.append((start, end + 1, entity_string, entity_info[2]))

        print("*=" * 10)
        print(f"Given input: {readable_input_str}")
        print(f"Model predict: {entity_lst}")
        # entity_lst is a list of (subtoken_start_pos, subtoken_end_pos, substring, entity_type)


if __name__ == '__main__':
    main()